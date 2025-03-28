# Copyright Axelera AI, 2024
import abc
import statistics
from typing import Dict, Generic, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


class AggregatorNoOp:
    """This class is used to represent an aggregator that does nothing

    This is used when you don't want to aggregate the results of an experiment
    """

    NAME = "No-Op"

    def __str__(self) -> str:
        return self.NAME

    @staticmethod
    def __call__(self) -> None:
        pass


class AggregatorBase(abc.ABC, Generic[T]):
    """Base class for the aggregators

    Must define the following variables:
        * NAME (str): a short memorable string identifying the aggregator. No spaces or characters
    """

    NAME: str

    @abc.abstractmethod
    def __call__(self, errors: Sequence[T]) -> float:
        pass


class AggregatorAverage(AggregatorBase):
    """Returns the average of the metric values"""

    NAME = "average"

    @staticmethod
    def __call__(errors: Sequence[float]) -> float:
        return sum(errors) / len(errors) if len(errors) else 0.0


class AggregatorStdev(AggregatorBase):
    """Returns the standard deviation of the metric values"""

    NAME = "stdev"

    @staticmethod
    def __call__(errors: Sequence[float]) -> float:
        return statistics.stdev(errors)


class AggregatorMax(AggregatorBase):
    """Returns the max of the metric values"""

    NAME = "max"

    @staticmethod
    def __call__(errors: Sequence[float]) -> float:
        return max(errors)


class AggregatorMin(AggregatorBase):
    """Returns the max of the metric values"""

    NAME = "min"

    @staticmethod
    def __call__(errors: Sequence[float]) -> float:
        return min(errors)


class AggregatorSum(AggregatorBase):
    """Returns the sum of the metric values"""

    NAME = "sum"

    @staticmethod
    def __call__(errors: Sequence[float]) -> float:
        return sum(errors)


class AggregatorMeanAveragePrecision(AggregatorBase):
    """Returns the mean average precision of the metric values

    The mean average precision is calculated as the average of the precision at each IoU threshold.

    The input is expected to have the mAP calculated at each IoU threshold.
    It should be a list of tuples
    with each entry (metric_name: str, metric_value: float)

    Example:
    ```
    metrics = [("map_50", 0.5), ("map_75", 0.75)]
    agg = AggregatorMeanAveragePrecision(threshold=0.5)
    value = agg(metrics)
    ```
    """

    def __init__(self, threshold: Optional[float] = None):
        assert threshold in [0.5, 0.75, None], "IoU threshold can be 0.5, 0.75 or None"
        self.threshold = threshold
        self.lookup = {None: "map", 0.5: "map_50", 0.75: "map_75"}

        if isinstance(self.threshold, float):
            self.NAME = f"mAP_{str(int(self.threshold*100))}"
        else:
            self.NAME = "mAP"

    def __call__(self, metrics: Sequence[Tuple[str, float]]) -> float:
        metrics_: Dict[str, float] = dict(metrics)

        return metrics_[self.lookup[self.threshold]]
