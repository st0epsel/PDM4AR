from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq
from inspect import CO_NESTED  # you may find this helpful

from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path, AdjacencyList
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        """
        Given the start and goal nodes, returns an ordered list of nodes from self.graph
        that make up the path between them, or an empty list if no path exists.
        """
        # Abstract function. Nothing to do here.
        pass


@dataclass
class UniformCostSearch(InformedGraphSearch):

    def sort_graph_weightwise(self) -> None:
        """Sorts the adjacency list of the graph in ascending order."""
        elements_sorted = {}
        for node, neighbors in self.graph.adj_list.items():
            elements_sorted[node] = sorted(list(neighbors), key=lambda n: self.graph.get_weight(node, n))

        self.graph.adj_list = elements_sorted
        return

    def path(self, start: X, goal: X) -> Path:

        # Initialize heapq Q with node start and cost_to_reach = 0
        Q = []
        heapq.heappush(Q, (0, start))  # (weight, node)

        cost_to_reach = {start: 0}  # Also stores all visited nodes V
        Parent = {start: None}
        while Q:
            ws, s = heapq.heappop(Q)  # ws = cost to reach s

            if s == goal:
                path = [s]
                while path[-1] != start:
                    path.append(Parent[path[-1]])
                path.reverse()
                return path

            for sprime in self.graph.adj_list[s]:
                new_cost_to_reach = ws + self.graph.get_weight(s, sprime)
                if sprime not in cost_to_reach or new_cost_to_reach < cost_to_reach[sprime]:
                    cost_to_reach[sprime] = new_cost_to_reach
                    Parent[sprime] = s
                    heapq.heappush(Q, (new_cost_to_reach, sprime))

        return []


@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0

    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def sort_graph_weightwise(self) -> None:
        """Sorts the adjacency list of the graph in ascending order."""
        elements_sorted = {}
        for node, neighbors in self.graph.adj_list.items():
            elements_sorted[node] = sorted(list(neighbors), key=lambda n: self.graph.get_weight(node, n))

        self.graph.adj_list = elements_sorted
        return

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        xu, yu = self.graph.get_node_coordinates(u)
        xv, yv = self.graph.get_node_coordinates(v)
        dx, dy = abs(xu - xv), abs(yu - yv)
        h = 0.8 * ((dx**2 + dy**2) ** 0.5)  # euclidian heuristic
        return h

    def path(self, start: X, goal: X) -> Path:

        graph = self.graph  # local reference (faster lookups)
        adj_list = graph.adj_list
        get_weight = graph.get_weight
        heuristic = self.heuristic

        # Initialize heapq Q with node start and cost_to_reach = 0
        Q = []
        heapq.heappush(Q, (0, start))  # (weight, node)

        cost_to_reach = {start: 0}  # Also stores all visited nodes V
        Parent = {start: None}
        visited = set()

        while Q:
            ws, s = heapq.heappop(Q)  # ws = cost to reach s

            # Skip if we've already processed this node with a lower cost
            if s in visited:
                continue
            visited.add(s)

            if s == goal:
                path = [s]
                while path[-1] != start:
                    path.append(Parent[path[-1]])
                path.reverse()
                return path

            for sprime in adj_list[s]:
                new_cost_to_reach = ws + get_weight(s, sprime)
                if sprime not in cost_to_reach or new_cost_to_reach < cost_to_reach[sprime]:
                    cost_to_reach[sprime] = new_cost_to_reach
                    Parent[sprime] = s
                    heapq.heappush(Q, (new_cost_to_reach + heuristic(sprime, goal), sprime))

        return []


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
