# Copyright Axelera AI, 2024
# Builds a dependency graph for the tasks in the pipeline

from collections import defaultdict
import enum
import functools
import sys

import networkx as nx

from .. import logging_utils, operators

LOG = logging_utils.getLogger(__name__)


# we now roughly define the network types as follows:
# - SINGLE_MODEL: a single task
# - CASCADE_NETWORK: a single cascade of tasks
# - PARALLEL_NETWORK: a parallel group of tasks
# - COMPLEX_NETWORK: any network that is not a simple cascade or parallel network
class NetworkType(enum.Enum):
    SINGLE_MODEL = enum.auto()
    CASCADE_NETWORK = enum.auto()
    PARALLEL_NETWORK = enum.auto()
    COMPLEX_NETWORK = enum.auto()


class DependencyGraph:
    def __init__(self, tasks):
        self.graph = defaultdict(list)
        self.task_map = {task.name: task for task in tasks}
        self.input_placeholder = "Input"
        self._initialize_graph(tasks)
        self.task_names = list(self.task_map.keys())
        self.model_names = [task.model_info.name for task in tasks]

    def _build_graph(self, tasks):
        for task in tasks:
            if isinstance(task.input, operators.InputFromROI):
                source_task_name = task.input.where
                self.graph[source_task_name].append(task.name)
            elif isinstance(task.input, (operators.Input, operators.InputWithImageProcessing)):
                self.graph[self.input_placeholder].append(task.name)

        # Ensure all tasks are in the graph, even if they have no dependencies
        for task in tasks:
            if task.name not in self.graph:
                self.graph[task.name] = []
        self.graph_nx = nx.from_dict_of_lists(self.graph, create_using=nx.DiGraph)

    def _check_task(self, task_name):
        if task_name not in self.task_names:
            if task_name in self.model_names:
                raise ValueError(f"Task {task_name} is a model, not a task")
            else:
                raise ValueError(f"Task {task_name} not found in the pipeline")

    def get_dependencies(self, task_name):
        return self.graph[task_name]

    @functools.lru_cache(maxsize=None)
    def get_master(self, task_name):
        self._check_task(task_name)
        predecessors = list(self.graph_nx.predecessors(task_name))
        if len(predecessors) > 1:
            raise ValueError("Unexpected network structure: multiple master nodes found")
        if not predecessors or predecessors[0] == self.input_placeholder:
            return None
        return predecessors[0]

    def clear_cache(self):
        self.get_master.cache_clear()

    def get_task(self, task_name):
        self._check_task(task_name)
        return self.task_map.get(task_name, None)

    def _initialize_graph(self, new_tasks):
        self._build_graph(new_tasks)
        self.clear_cache()

    def print_graph(self, stream=sys.stdout):
        indent_char = "  "  # Indentation character
        branch_char = "│ "  # Branch character
        arrow_char = "└─"  # Arrow character

        # Create a mapping of nodes to their respective layers
        layers = {}
        for task_name in self.task_map:
            if task_name == self.input_placeholder:
                layers[task_name] = 0
            else:
                if dependencies := self.get_dependencies(task_name):
                    layer = max(layers.get(dep, 0) for dep in dependencies) + 1
                else:
                    layer = 1
                layers[task_name] = layer

        max_layer = max(layers.values())
        visited = set()

        def print_dependencies(task, level, prefix=""):
            if task in visited:
                return
            visited.add(task)

            if level > 0:
                prefix += indent_char * (level - 1)
                if level == max_layer:
                    prefix += arrow_char
                else:
                    prefix += branch_char

            print(f"{prefix}{task}", file=stream)

            dependencies = self.get_dependencies(task)
            if dependencies:
                for i, dep in enumerate(dependencies):
                    if i == len(dependencies) - 1:
                        print_dependencies(dep, level + 1, prefix + "  ")
                    else:
                        print_dependencies(dep, level + 1, prefix + "│ ")

        print_dependencies(self.input_placeholder, 0)

    def visualize_graph(self):
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        self.input_placeholder = "Input"  # Placeholder for the input node

        # Create a mapping of nodes to their respective layers
        layers = {}
        for task_name in self.task_map:
            if task_name == self.input_placeholder:
                layers[task_name] = 0
            else:
                dependencies = self.get_dependencies(task_name)
                if dependencies:
                    layer = max(layers.get(dep, 0) for dep in dependencies) + 1
                else:
                    layer = 1
                layers[task_name] = layer

        # Add edges to the graph and assign subset (layer) attribute to each node
        for source, destinations in self.graph.items():
            for dest in destinations:
                G.add_edge(source, dest)
                G.nodes[source]["subset"] = layers.get(source, 0)  # Use get() with default value 0
                G.nodes[dest]["subset"] = layers.get(dest, 0)  # Use get() with default value 0

        # Set the positions of nodes using multipartite_layout
        pos = nx.multipartite_layout(G, subset_key="subset")
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1000,
            node_color='lightblue',
            font_size=12,
            arrows=True,
        )
        plt.axis('off')
        plt.show()

    @property
    def network_type(self):
        if len(self.task_map) == 1:
            return NetworkType.SINGLE_MODEL
        elif self._is_cascade_network():
            return NetworkType.CASCADE_NETWORK
        elif self._is_parallel_network():
            return NetworkType.PARALLEL_NETWORK
        else:
            return NetworkType.COMPLEX_NETWORK

    def _is_cascade_network(self):
        if not nx.is_directed_acyclic_graph(self.graph_nx):
            return False
        # Check if there's a single path from input to output
        roots = [n for n in self.graph_nx.nodes() if self.graph_nx.in_degree(n) == 0]
        leaves = [n for n in self.graph_nx.nodes() if self.graph_nx.out_degree(n) == 0]
        return (
            len(roots) == 1
            and len(leaves) == 1
            and nx.has_path(self.graph_nx, roots[0], leaves[0])
        )

    def _is_parallel_network(self):
        # A parallel network should have all tasks directly connected to the input
        # and no connections between tasks
        input_nodes = [node for node in self.graph_nx.nodes() if node == self.input_placeholder]
        if len(input_nodes) != 1:
            return False
        input_node = input_nodes[0]
        return all(
            self.graph_nx.has_edge(input_node, node)
            for node in self.graph_nx.nodes()
            if node != input_node
        ) and all(
            self.graph_nx.out_degree(node) == 0
            for node in self.graph_nx.nodes()
            if node != input_node
        )

    def get_root_and_leaf_tasks(self):
        if self.network_type == NetworkType.SINGLE_MODEL:
            task = list(self.task_map.keys())[0]
            return task, task  # For single model, root and leaf are the same
        elif self.network_type == NetworkType.CASCADE_NETWORK:
            # Find the root node (first task after 'Input')
            root_nodes = [
                node
                for node in self.graph_nx.nodes()
                if self.graph_nx.in_degree(node) == 1
                and list(self.graph_nx.predecessors(node))[0] == 'Input'
            ]
            # Find the leaf node (no outgoing edges)
            leaf_nodes = [
                node for node in self.graph_nx.nodes() if self.graph_nx.out_degree(node) == 0
            ]
            if len(root_nodes) == 1 and len(leaf_nodes) == 1:
                return root_nodes[0], leaf_nodes[0]
            else:
                raise ValueError(
                    "Unexpected network structure: multiple or no root/leaf nodes found"
                )
        else:
            raise ValueError("Unsupported network type")
