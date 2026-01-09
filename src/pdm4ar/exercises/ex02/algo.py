from abc import abstractmethod, ABC

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes
import networkx as nx
from networkx import random_geometric_graph, DiGraph


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, [] if a path does not exist
        """
        pass

    def sort_graph(self, graph: AdjacencyList, reversed=False) -> AdjacencyList:
        """Sorts the adjacency list of the graph in ascending order."""
        elements_sorted = {}
        for node, neighbors in graph.items():
            elements_sorted[node] = sorted(neighbors, reverse=reversed)

        return elements_sorted


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # print(f"Graph:\n {graph}")
        # print(f"Start: {start}, Goal: {goal}\n")
        graph = self.sort_graph(graph, reversed=True)
        Q = [start]  # Q is the Queue of active nodes
        V = []  # V is the list of opened nodes
        Vset = {start}
        Parent = {}  # Parent is a mapping from node to its parent node
        while Q:
            # print(f"Q: {Q},  V: {V}")
            s = Q.pop()  # E = Expand Node
            V.append(s)
            # print(f"Q: {Q},  V: {V}, -> s: {s}")

            # If Goal Node is reached, reconstruct path and return
            if s == goal:
                path = [s]
                while path[-1] != start:
                    path.append(Parent[path[-1]])
                path.reverse()
                # print(f"path: {path}")
                return path, V

            # Iterate over all child nodes s' of s

            for sprime in graph[s]:
                # print(f"s': {sprime}")
                if sprime not in Vset:
                    Q.append(sprime)
                    Vset.add(sprime)
                    Parent[sprime] = s

        # print(f"Q: {Q},  \nV: {V}")
        return [], V


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        graph = self.sort_graph(graph)
        # print(f"Graph:\n {graph}")
        # print(f"Start: {start}, Goal: {goal}\n")
        Q = [start]  # Q is the Queue of active nodes
        V = []  # V is the list of opened nodes
        Vset = {start}
        Parent = {}  # Parent is a mapping from node to its parent node
        while Q:
            # print(f"Q: {Q},  V: {V}")
            s = Q.pop(-1)  # E = Expand Node
            V.append(s)
            # print(f"Q: {Q},  V: {V}, -> s: {s}")

            # If Goal Node is reached, reconstruct path and return
            if s == goal:
                path = [s]
                while path[-1] != start:
                    path.append(Parent[path[-1]])
                path.reverse()
                # print(f"path: {path}")
                return path, V

            # Iterate over all child nodes s' of s

            for sprime in graph[s]:
                # print(f"s': {sprime}")
                if sprime not in Vset:
                    Vset.add(sprime)
                    Q.insert(0, sprime)
                    Parent[sprime] = s

        # print(f"Q: {Q},  \nV: {V}")
        return [], V


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        graph = self.sort_graph(graph, reversed=True)
        # print(f"Graph:\n {graph}")
        # print(f"Start: {start}, Goal: {goal}\n")
        d = 1  # depth
        Vlast = None
        while True:
            # print(f"\n--- d = {d} ---")
            Q = [start]  # Q is the Queue of active nodes
            V = []  # V is the list of opened nodes
            D = {start: 1}  # D is the depth of each node
            Vset = {start}
            Parent = {}  # Parent is a mapping from node to its parent node
            while Q:
                # print(f"Q: {Q}, D: {D},   V: {V}")
                s = Q.pop(0)  # E = Expand Node
                V.append(s)
                # print(f"Q: {Q}, D: {D},   V: {V}")

                # If Goal Node is reached, reconstruct path and return
                if s == goal:
                    path = [s]
                    while path[-1] != start:
                        path.append(Parent[path[-1]])
                    path.reverse()
                    return path, V

                # Iterate over all child nodes s' of s
                for sprime in graph[s]:
                    # print(f"s': {sprime}")
                    if (sprime not in Vset) and (D[s] < d):
                        Parent[sprime] = s
                        Vset.add(sprime)
                        D[sprime] = D[s] + 1
                        Q.insert(0, sprime)

            # print(f"Q: {Q},  \nV: {V}")
            if V == Vlast:
                break
            Vlast = V
            d += 1
        return [], V


if __name__ == "__main__":
    # easy01: AdjacencyList = random_geometric_graph(10, 0.5, seed=9)
    # easy01: AdjacencyList = {1: {2, 3}, 2: {3, 4}, 3: {4}, 4: {3}, 5: {6}, 6: {3}}
    easy01: AdjacencyList = {
        9: {1, 4},
        0: {2, 4, 5, 6, 7, 8},
        1: {9, 2, 4, 5},
        2: {0, 1, 5, 8},
        3: {6, 7},
        4: {0, 1, 6, 7, 9},
        5: {0, 1, 2, 8},
        6: {0, 3, 4, 7},
        7: {0, 3, 4, 6, 8},
        8: {0, 2, 5, 7},
    }

    k = IterativeDeepening()

    print(f"easy01:        {easy01}\neasy01 sorted: {k.sort_graph(easy01)}")
    # print(k.search(easy01, 5, 2))
