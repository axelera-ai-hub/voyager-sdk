# Copyright Axelera AI, 2024
# Builds a dependency graph for the tasks in the pipeline

from collections import defaultdict
import enum
import functools

import networkx as nx

from axelera import types

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


class EdgeType(enum.Enum):
    EXECUTION = enum.auto()  # For execution flow
    RESULT = enum.auto()  # For result organization


class DependencyGraph:
    def __init__(self, tasks):
        self.graph = defaultdict(list)
        self.result_graph = defaultdict(list)  # New graph for result organization
        self.task_map = {task.name: task for task in tasks}
        self.input_placeholder = "Input"
        self.task_names = list(self.task_map.keys())
        self._initialize_graph(tasks)
        self.model_names = [task.model_info.name for task in tasks]

        # Build result view automatically
        self._build_result_graph(tasks)

    def _build_graph(self, tasks):
        for task in tasks:
            if task.is_dl_task:
                if isinstance(task.input, operators.InputFromROI):
                    source_task_name = task.input.where
                    self.graph[source_task_name].append(task.name)
                elif isinstance(task.input, (operators.Input, operators.InputWithImageProcessing)):
                    self.graph[self.input_placeholder].append(task.name)
            elif TrackerHelper.is_tracker_task(task):
                # Use TrackerHelper to handle tracker dependencies
                dependencies = TrackerHelper.get_tracker_dependencies(task)

                # Add edge from detection task to tracker
                self.graph[dependencies['detection']].append(task.name)

                # Add edge from embeddings task to tracker if it exists
                if 'embeddings' in dependencies:
                    embeddings_task_name = dependencies['embeddings']
                    self.graph[embeddings_task_name].append(task.name)
            else:
                raise ValueError(
                    f"Handle this case: Task {task.name} is not a DL task or a tracker task"
                )

        # Ensure all tasks are in the graph, even if they have no dependencies
        for task in tasks:
            if task.name not in self.graph:
                self.graph[task.name] = []

        # Create NetworkX graph for execution view
        self.graph_nx = nx.from_dict_of_lists(self.graph, create_using=nx.DiGraph)

    def _is_internal_task(self, task_name):
        return task_name == 'axelera-tiles-internal'

    def _check_task(self, task_name):
        # This is a special internal task used for tiling
        if self._is_internal_task(task_name):
            return
        if task_name not in self.task_names:
            if task_name in self.model_names:
                raise ValueError(f"Task {task_name} is a model, not a task")
            else:
                raise ValueError(f"Task {task_name} not found in the pipeline")

    def _build_result_graph(self, tasks):
        # Start with a copy of the execution graph
        self.result_graph = {k: v.copy() for k, v in self.graph.items()}

        # Find tracker nodes and handle them specially
        for task in tasks:
            if TrackerHelper.is_tracker_task(task):
                # Use the centralized helper to modify the result graph
                self.result_graph = TrackerHelper.handle_tracker_result_view(
                    task, self.graph, self.result_graph
                )

        # Create NetworkX graph for result view
        self.result_graph_nx = nx.from_dict_of_lists(self.result_graph, create_using=nx.DiGraph)

    def get_dependencies(self, task_name, view=EdgeType.EXECUTION):
        """Get task dependencies based on the specified view."""
        if view == EdgeType.EXECUTION:
            return self.graph[task_name]
        else:
            return self.result_graph[task_name]

    @functools.lru_cache(maxsize=None)
    def get_master(self, task_name, view=EdgeType.EXECUTION):
        """Get the master (predecessor) task based on the specified view.

        Args:
            task_name: The name of the task to find the master for
            view: Which graph view to use (EXECUTION or RESULT)

        Returns:
            The name of the master task, or None if there is no master
        """
        if self._is_internal_task(task_name):
            return None
        self._check_task(task_name)

        # Use the appropriate graph based on the view
        graph_to_use = self.graph_nx if view == EdgeType.EXECUTION else self.result_graph_nx

        predecessors = list(graph_to_use.predecessors(task_name))

        # Handle special case for trackers
        if len(predecessors) > 1:
            task = self.task_map.get(task_name)
            if TrackerHelper.is_tracker_task(task):
                # For tracker tasks, consider only the bbox_task_name as the master
                dependencies = TrackerHelper.get_tracker_dependencies(task)
                if 'detection' in dependencies and dependencies['detection'] in predecessors:
                    return dependencies['detection']
            # For other tasks with multiple predecessors, raise an error
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
        self._build_result_graph(new_tasks)  # This will create both graph representations
        self.clear_cache()

    def print_graph(self, log=print, view=EdgeType.EXECUTION):
        """Print the graph in the specified view.

        Args:
            stream: Either a file-like object with a write method, or a logging function
            view: Which graph view to print (EXECUTION or RESULT)
        """
        graph_to_use = self.graph if view == EdgeType.EXECUTION else self.result_graph

        log(f"\n--- {view.name} VIEW ---")
        indent_char = "  "  # Indentation character
        branch_char = "│ "  # Branch character
        arrow_char = "└─"  # Arrow character

        # Create a mapping of nodes to their respective layers
        layers = {}
        for task_name in self.task_map:
            if task_name == self.input_placeholder:
                layers[task_name] = 0
            else:
                if deps := [d for d in graph_to_use.get(task_name, []) if d in graph_to_use]:
                    layer = max(layers.get(dep, 0) for dep in deps) + 1
                else:
                    layer = 1
                layers[task_name] = layer

        max_layer = max(layers.values()) if layers else 0
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

            output_line = f"{prefix}{task}"
            log(output_line)

            dependencies = graph_to_use.get(task, [])
            if dependencies:
                for i, dep in enumerate(dependencies):
                    if i == len(dependencies) - 1:
                        print_dependencies(dep, level + 1, prefix + "  ")
                    else:
                        print_dependencies(dep, level + 1, prefix + "│ ")

        print_dependencies(self.input_placeholder, 0)

    def print_all_views(self, log=print):
        """Print both execution and result views of the graph.

        Args:
            stream: Either a file-like object or a logging function
        """
        self.print_graph(log=log, view=EdgeType.EXECUTION)
        self.print_graph(log=log, view=EdgeType.RESULT)

    def visualize_graph(self, view=EdgeType.EXECUTION):
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        self.input_placeholder = "Input"  # Placeholder for the input node

        graph_to_use = self.graph if view == EdgeType.EXECUTION else self.result_graph

        # Create a mapping of nodes to their respective layers
        layers = {}
        for task_name in self.task_map:
            if task_name == self.input_placeholder:
                layers[task_name] = 0
            else:
                dependencies = graph_to_use.get(task_name, [])
                if dependencies:
                    layer = max(layers.get(dep, 0) for dep in dependencies) + 1
                else:
                    layer = 1
                layers[task_name] = layer

        # Add edges to the graph and assign subset (layer) attribute to each node
        for source, destinations in graph_to_use.items():
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
        if self._is_tracker_cascade():
            return True
        # Check if there's a single path from input to output
        roots = [n for n in self.graph_nx.nodes() if self.graph_nx.in_degree(n) == 0]
        leaves = [n for n in self.graph_nx.nodes() if self.graph_nx.out_degree(n) == 0]
        return (
            len(roots) == 1
            and len(leaves) == 1
            and nx.has_path(self.graph_nx, roots[0], leaves[0])
        )

    def _is_tracker_cascade(self):
        """
        Check if the network follows the special tracker cascade pattern with detection and re-ID models
        feeding into a tracker.
        """
        return TrackerHelper.is_tracker_cascade(
            self.graph_nx, self.task_map, self.input_placeholder
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

            # Handle tracker special case
            if not (len(root_nodes) == 1 and len(leaf_nodes) == 1):
                # Check if this is a tracker cascade
                tracker_candidates = [
                    task_name
                    for task_name in self.task_map
                    if TrackerHelper.is_tracker_task(self.task_map[task_name])
                ]

                if tracker_candidates:
                    # Find a tracker that's a leaf node
                    tracker_leaves = [t for t in tracker_candidates if t in leaf_nodes]
                    if tracker_leaves:
                        # For tracker cascades, use the first root node and the tracker leaf
                        tracker_leaf = tracker_leaves[0]
                        if root_nodes:
                            return root_nodes[0], tracker_leaf

            if len(root_nodes) == 1 and len(leaf_nodes) == 1:
                return root_nodes[0], leaf_nodes[0]
            else:
                raise ValueError(
                    "Unexpected network structure: multiple or no root/leaf nodes found"
                )
        else:
            raise ValueError("Unsupported network type")


class TrackerHelper:
    """Helper class for handling tracker-specific functionality in the dependency graph."""

    @staticmethod
    def is_tracker_task(task):
        """Check if the given task is a tracker task."""
        return (
            not task.is_dl_task
            and task.model_info.task_category == types.TaskCategory.ObjectTracking
        )

    @staticmethod
    def get_tracker_dependencies(task):
        """Get the detection and optional embeddings task names that a tracker depends on."""
        dependencies = {}
        if hasattr(task.cv_process[0], 'bbox_task_name'):
            dependencies['detection'] = task.cv_process[0].bbox_task_name

        if (
            hasattr(task.cv_process[0], 'embeddings_task_name')
            and task.cv_process[0].embeddings_task_name
        ):
            dependencies['embeddings'] = task.cv_process[0].embeddings_task_name

        return dependencies

    @staticmethod
    def handle_tracker_result_view(tracker_task, graph, result_graph):
        """
        Modify the result view graph to show trackers directly connected to input.
        This reflects that trackers produce standalone results regardless of their inputs.
        """
        # Find all source tasks that the tracker depends on
        for source_task, dependent_tasks in graph.items():
            if tracker_task.name in dependent_tasks:
                # Remove dependency from the result view
                if source_task != 'Input':  # Only modify if not already from input
                    if tracker_task.name in result_graph.get(source_task, []):
                        result_graph[source_task].remove(tracker_task.name)

                    # Add direct connection from input to tracker
                    if 'Input' not in result_graph:
                        result_graph['Input'] = []
                    if tracker_task.name not in result_graph['Input']:
                        result_graph['Input'].append(tracker_task.name)

        return result_graph

    @staticmethod
    def is_tracker_cascade(graph_nx, task_map, input_placeholder='Input'):
        """
        Check if the network follows the special tracker cascade pattern with:
        - Detection model feeding into a tracker
        - Re-ID model feeding into the same tracker (optional)
        - A clear execution flow
        """
        # Find nodes that might be trackers
        tracker_candidates = [
            task_name
            for task_name, task in task_map.items()
            if task.model_info.task_category == types.TaskCategory.ObjectTracking
        ]

        if not tracker_candidates:
            return False

        # For each tracker candidate, check if it has appropriate inputs
        for tracker in tracker_candidates:
            task = task_map.get(tracker)
            if not task:
                continue

            # Get the detection task name this tracker depends on
            bbox_task_name = task.cv_process[0].bbox_task_name

            # Get embeddings task name if it exists
            embeddings_task_name = None
            if hasattr(task.cv_process[0], 'embeddings_task_name'):
                embeddings_task_name = task.cv_process[0].embeddings_task_name

            # Get all predecessors of the tracker
            predecessors = list(graph_nx.predecessors(tracker))

            # Check if this is a valid tracker cascade:
            # 1. Tracker must have the bbox_task_name as a predecessor
            # 2. If embeddings_task_name is specified, it should also be a predecessor
            # 3. The graph must be a directed acyclic graph (DAG)

            # Tracker should depend on detection task
            if bbox_task_name not in predecessors:
                continue

            # If embeddings task is specified, make sure it's properly connected
            if embeddings_task_name and embeddings_task_name not in predecessors:
                continue

            # Check if the graph structure without the tracker is a valid DAG
            nodes_without_tracker = [
                n for n in graph_nx.nodes() if n != tracker and n != input_placeholder
            ]
            subgraph = graph_nx.subgraph(nodes_without_tracker)
            del subgraph  # We dont need the subgraph, it was just to check DAGness

            # It's a tracker cascade if:
            # - The tracker is a leaf node (no outgoing edges)
            # - The subgraph without the tracker is either a cascade or a parallel network
            if graph_nx.out_degree(tracker) == 0:
                # Check if subgraph is a simple path or parallel structure
                if len(nodes_without_tracker) <= 2:
                    # Simple case: just one or two nodes before the tracker
                    return True

                # For more complex cases, check if the predecessors form a valid structure
                # All predecessors must be connected to input directly or indirectly
                input_node = input_placeholder
                all_connected_to_input = all(
                    nx.has_path(graph_nx, input_node, pred) for pred in predecessors
                )

                if all_connected_to_input:
                    # If the graph is a pure cascade or has a valid cascaded structure
                    # where all paths eventually lead to the tracker
                    return True

        return False
