# Copyright Axelera AI, 2024
# Evaluator implementation for classification tasks.
from __future__ import annotations

import abc
import collections
import heapq
import re
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from axelera import types
from axelera.app.meta import ClassificationMeta
from axelera.app.model_utils.box import xyxy2xywh
from axelera.app.model_utils.segment import simple_resize_masks
from axelera.app.utils import logging_utils

from .statistic_aggregator import (
    AggregatorAverage,
    AggregatorBase,
    AggregatorMax,
    AggregatorMin,
    AggregatorSum,
)

LOG = logging_utils.getLogger(__name__)

_RE_TOPK = re.compile(r"top(\d+)_([a-z]+)")

_aggregators = {
    "avg": AggregatorAverage,
    "sum": AggregatorSum,
    "min": AggregatorMin,
    "max": AggregatorMax,
}


class GeneratorBase(collections.abc.Iterable):
    """This class is to put common functionality for all the generators

    Each deriving class should define a functionality

    def __iter__(self):
        which yields a tuple (prediction, label)

    And attributes:
        * NAME (str): a unique identifier for the generator
    """

    NAME: str


PT = TypeVar("PT")  # prediction type
LT = TypeVar("LT")  # label type


class GeneratorFull(GeneratorBase, Generic[PT, LT]):
    """Generator that takes a full set of predictions and inputs"""

    NAME = "generator-full"

    def __init__(self, predictions: Sequence[PT], labels: Sequence[LT]) -> None:
        assert len(predictions) == len(labels)
        self.predictions = predictions
        self.labels = labels

    def __iter__(self):
        for prediction, label in zip(self.predictions, self.labels):
            yield (prediction, label)


class Comparable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __lt__(self, other: Any) -> bool:
        ...


CT = TypeVar("CT", bound=Comparable)


def _top_k_metric(predicted: Sequence[CT], actual: int, k: int) -> float:
    assert k > 0
    topk = heapq.nlargest(k, predicted, key=predicted.get)
    return float(actual in topk)


class MetricNoOp:
    NAME = "no-op"

    def __call__(self) -> None:
        """Does nothing

        Returns:
            NoReturn: nothing
        """
        pass

    def __eq__(self, other: Any) -> bool:
        return self.NAME == other

    def __hash__(self) -> int:
        return hash(self.NAME)


class MetricBase(abc.ABC):
    """Base class for metrics which evaluate the accuracy of a prediction compared to a label

    Must define the following variables:

        * NAME (str): a short memorable string identifying the metric. No spaces or characters

    Args:
        predicted: a prediction from the model
        actual: a label from the validation set

    Returns:
        a value that indicates the accuracy of the prediction

    """

    NAME: str

    @abc.abstractmethod
    def __call__(self, predicted: Any, actual: Any) -> Any:
        pass

    def __eq__(self, other) -> bool:
        return self.NAME == other

    def __hash__(self) -> int:
        return hash(self.NAME)


class MetricClassificationTopK(MetricBase):
    """The top-k metric for classification problems
    Args:
        k: the number of classes to be considered
    """

    NAME = "accuracy-top-k"

    def __init__(self, k: int) -> None:
        """Constructs the top-k metric"""
        self._k = k
        self.NAME = "accuracy-top-{:d}".format(k)

    def __call__(self, predicted: Sequence[CT], actual: int) -> float:
        """Gets the top-k accuracy of a classification prediction

        Args:
            predicted: a list of values indicating the likeliness of the predicted class
            actual: the true class number

        Returns:
            1 if actual class is one of the top-k likely predicted classes
        """
        return _top_k_metric(predicted, actual, self._k)


def _classification_metrics(metric: str):
    if match := _RE_TOPK.match(metric):
        k, agg = match.groups()
        return MetricClassificationTopK(k=int(k)), _aggregators[agg]()
    else:
        raise ValueError(f"Unknown classification metric {metric}")


def _eval_map_generator(
    generator: GeneratorBase, calculate_metric: Callable[..., float]
) -> List[float]:
    """Calls the calculate metric for all elements in the validation set

    This variant is lazy and gets predictions and labels from a generator

    Arguments:
        predictions: A list of list of predictions. Each prediction is a list of class
            probabilities
        labels: A single value with the true class number
        calculate_metric: a function that takes a list of predictions and a class number
        and outputs a metric value
    Returns:
        A list of metric values for all N elements in the validation set
    """
    return [calculate_metric(prediction, label) for prediction, label in generator]


class MetricDefinitions(object):
    """Stores the metrics and aggregators that will be computed during an evaluation

    An evaluation can compute several values that compare a prediction to a label or reference,
    using several respective metrics. For example, one might calculate the Top5 accuracy as well
    as the Top1 accuracy which are two separate metrics.

    For every given metric, one might also aggregate the values in different ways. Taking the
    average, or the sum are two possibilities.

    The metric definitions is a tree with the first leaves representing the metrics and the second
    level of leaves the aggregators. The tree below is one example.

        Top1                    Top5
      |     |              |     |      |
    Average Sum         Average   Min   Max

    Optionally one, can already supply a list of metrics and aggregators.
    The list of metrics and aggregators must be the same length. There may be several
    repeated values in metrics if a metric has several aggreators. The order doesn't matter
    as long as the correspondence is right. For the example in the class
    docstring, the lists would be

    metrics = [(MetricClassificationTopK(k=1), MetricClassificationTopK(k=1),
                MetricClassificationTopK(k=5), MetricClassificationTopK(k=5),
                MetricClassificationTopK(k=5)]
    aggregators = [(AggregatorAverage(), AggregatorSum(),
                AggretorAverage(), AggregatorMin(),
                AggregatorMax()]

    Args:
        metrics: An optional list of metrics to use.
        aggregators: An optional list of aggregators
    """

    def __init__(
        self,
        metrics: Optional[Sequence[MetricBase]] = None,
        aggregators: Optional[Sequence[AggregatorBase]] = None,
    ) -> None:
        """Initializes the metric definitions"""

        self._d: Dict[MetricBase, List[AggregatorBase]] = collections.defaultdict(list)

        if (metrics is not None) and (aggregators is not None):
            if len(metrics) == len(aggregators):
                self.add_several(metrics, aggregators)
            else:
                raise ValueError("Number of metrics and aggregators must be equal")

    def add(self, metric: MetricBase, aggregator: AggregatorBase) -> None:
        """Adds an aggregator to a metric
        Note that the metric can have several aggregators.
        Args:
            metric: The metric to add the aggregator to.
            aggregator: The aggregator to add to the metric
        """

        self._d[metric].append(aggregator)

    def add_several(
        self, metrics: Sequence[MetricBase], aggregators: Sequence[AggregatorBase]
    ) -> None:
        """Adds several metrics and aggregators

        The list of metrics and aggregators must be the same length. There may be several
        repeated values in metrics if a metric has several aggreators. The order doesn't matter
        as long as the correspondence is right. For the example in the class
        docstring, the lists would be

        metrics = [(MetricClassificationTopK(k=1), MetricClassificationTopK(k=1),
                    MetricClassificationTopK(k=5), MetricClassificationTopK(k=5),
                    MetricClassificationTopK(k=5)]
        aggregators = [(AggregatorAverage(), AggregatorSum(),
                    AggretorAverage(), AggregatorMin(),
                    AggregatorMax()]

        Args:
            metrics: An optional list of metrics to use.
            aggregators: An optional list of aggregators
        """

        if len(metrics) != len(aggregators):
            raise ValueError("Number of metrics and aggregators must be equal")

        self._d = collections.defaultdict(list)

        for metric, aggregator in zip(metrics, aggregators):
            self._d[metric].append(aggregator)

    def items(self):
        """Returns a list of tuples (metric, aggregator)"""
        return self._d.items()

    def num_metrics(self) -> int:
        """Returns the unique number of metrics"""
        return len(self._d.keys())

    def get_metrics(self) -> List[MetricBase]:
        """Returns a list of all metrics"""
        return list(self._d.keys())

    def num_aggregators(self, metric: MetricBase) -> int:
        """Returns the number of aggregators associated with a metric"""
        return len(self._d[metric])

    def get_aggregators(self, metric: MetricBase) -> List[AggregatorBase]:
        """Returns the aggregators associated with a metric"""
        return self._d[metric]

    def __repr__(self) -> str:
        """Returns a string representation of the metrics and aggregators"""
        return str(self._d)


class EvaluatorBase(object):
    """An evaluator combines a generator, a metric, and an aggregator

    Args:
        generator: the generator that outputs samples of the form (prediction, label)
        calculate_metrics: the metric function that evaluates the accuracy of a prediction
            to a label
        aggregate_metrics: the function that combines all metric values to one
    """

    def __init__(self, generator: GeneratorBase, metric_definitions: MetricDefinitions) -> None:
        """Constructs an Evaluator from the generator, metric, aggregator triple"""
        self.generator = generator
        if not metric_definitions.items():
            raise ValueError("No metrics defined")

        self._metric_defs = metric_definitions

    @property
    def metric_definitions(self) -> MetricDefinitions:
        """Returns the metrics and aggregators that are computed by this evaluator

        Returns:
            The combinations of metric and aggregators
        """
        return self._metric_defs

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute the aggregated metric values

        An example is the mean square error, where the square error would be a metric and the mean
        is an aggregator

        Returns:
            a dict with the key is the name of the aggregator and the value is the
            aggregated metric value
        """
        d: DefaultDict[str, Dict[str, float]] = collections.defaultdict(dict)
        for metric, aggregators in self._metric_defs.items():
            errors = _eval_map_generator(self.generator, metric)
            for agg in aggregators:
                d[metric.NAME][agg.NAME] = agg(errors)

        return d


class EvaluatorFull(EvaluatorBase):
    """Builds an Evaluator that works on a full dataset of predictions and labels

    This variant assumes that all predictions have been calculated and are available in memory.
    This can be more optimal than running a lazy version such as EvaluatorBase, since
    predictions can be made in batches

    Arguments:
        predictions: a sequence of predictions
        labels: a sequence of labels of the same size as predictions
        calculate_metric: the metric function that evaluates the accuracy of a prediction to a
            label
        aggregate_metric: the function that combines all metric values to one
    """

    def __init__(
        self, predictions: Any, labels: Any, metric_definitions: MetricDefinitions
    ) -> None:
        generator = GeneratorFull(predictions, labels)
        super().__init__(generator, metric_definitions)


USE_OFFLINE_EVAL = False


def create_classification_metric_definitions(top_k: int) -> MetricDefinitions:
    metric_defs = MetricDefinitions()
    metric_map = dict()
    metrics = ["top1_avg"] if top_k == 1 else ["top1_avg", f"top{top_k}_avg"]
    for metric in metrics:
        metric_map[metric] = _classification_metrics(metric)
        metric_defs.add(*metric_map[metric.lower()])
    return metric_defs


class ClassificationEvaluator(types.Evaluator):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.metric_defs = create_classification_metric_definitions(top_k)
        if USE_OFFLINE_EVAL:
            self.predictions = []
            self.labels = []
        else:
            self.metric_results = collections.defaultdict(list)
        self._has_confirmed_topk = False
        self._default_topk = top_k

    def process_meta(self, meta: ClassificationMeta):
        prediction = meta.to_evaluation().data
        label = meta.access_ground_truth().data
        assert (
            meta.num_classes != 1
        ), "Please set num_classes in the YAML models section for classification tasks and redeploy the model"
        if not self._has_confirmed_topk:
            _, scores = meta.get_result(0)
            if len(scores) != self._default_topk:
                LOG.info(f"Setting measurement of topk to {len(scores)}")
                self.metric_defs = create_classification_metric_definitions(len(scores))
            self._has_confirmed_topk = True

        if USE_OFFLINE_EVAL:
            self.predictions.append(prediction)
            self.labels.append(label)
        else:
            for metric in self.metric_defs.get_metrics():
                result = metric(prediction, label)
                self.metric_results[metric.NAME].append(result)

    def collect_metrics(self):
        if USE_OFFLINE_EVAL:
            evaluator = EvaluatorFull(self.predictions, self.labels, self.metric_defs)
            results = evaluator.compute()
        else:
            results = collections.defaultdict(dict)
            for metric, aggregators in self.metric_defs.items():
                for aggregator in aggregators:
                    results[metric.NAME][aggregator.NAME] = aggregator(
                        self.metric_results[metric.NAME]
                    )

        metric_names = list(results.keys())
        aggregator_dict = {metric: list(aggs.keys()) for metric, aggs in results.items()}

        eval_result = types.EvalResult(
            metric_names,
            aggregator_dict,
            'accuracy-top-1',
            'average',
        )
        for metric_name, aggregators in results.items():
            for agg_name, value in aggregators.items():
                eval_result.set_metric_result(metric_name, value, agg_name, is_percentage=True)
        return eval_result
