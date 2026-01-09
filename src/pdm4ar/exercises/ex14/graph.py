from operator import is_
import numpy as np
import networkx as nx
from typing import Sequence, Tuple, List, Optional, Dict
from numpydantic import NDArray
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, nearest_points
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree

from pdm4ar.exercises_def.structures import out_dir
from pdm4ar.exercises.ex14.global_config import GlobalConfig
from pdm4ar.exercises.ex14.helper_fcts import plot_graph

from dg_commons.sim.models.obstacles import StaticObstacle


class GraphBuilder:

    def __init__(
        self,
        obstacles: Sequence[StaticObstacle],
        robot_radius: float,
        safety_margin: float = 0.0,
        n_samples: int = 500,
        node_connection_mult: float = 1.5,
        goals: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        collections: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        initial_sates: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        """
        Docstring for __init__
        """
        self.VERBOSE = GlobalConfig.GRAPH_VERBOSE
        self.id = id

        self.robot_radius = robot_radius
        self.inflation_dist = robot_radius + safety_margin

        self.obstacles = obstacles

        if goals:
            self.goals = goals
        else:
            self.goals = {}
        self.goal_pos = [[row[0], row[1]] for row in self.goals.values()]

        if collections:
            self.collections = collections
        else:
            self.collections = {}
        self.collections_pos = [[row[0], row[1]] for row in self.collections.values()]

        if initial_sates:
            self.initial_states = initial_sates
        else:
            self.initial_states = {}
        self.initial_state_pos = [[row[0], row[1]] for row in self.initial_states.values()]

        self.num_pois = len(self.goals) + len(self.collections) + len(self.initial_states)
        if self.VERBOSE:
            print(f"Number of POIs (goals {len(self.goals)} + collections {len(self.collections)}): {self.num_pois}")

        self.free_space = self._calc_free_space(inflation_dist=0.0)

        self.traversable_space = self._calc_free_space(inflation_dist=self.inflation_dist)

        # Extract outer boundary shape
        raw_boundary = obstacles[-1].shape
        if isinstance(raw_boundary, LinearRing):
            raw_boundary = Polygon(raw_boundary)
        self.min_x, self.min_y, self.max_x, self.max_y = raw_boundary.bounds
        area = (self.max_x - self.min_x) * (self.max_y - self.min_y)  # m^2
        self.node_dist = np.sqrt(area / n_samples)

        self.n_samples = n_samples

        self.connection_radius = self.node_dist * node_connection_mult

        self.G = nx.Graph()

        if self.VERBOSE:
            print(f"   optimal node dist = {self.node_dist}")
            print(f"   connection_radius = {self.connection_radius}")

        self._generate_graph()

    def _generate_graph(self):
        G = self.G
        connection_radius = self.connection_radius
        traversable_space = self.traversable_space

        # Genearate Nodes
        nodes = tuple(map(tuple, self._calc_grid_nodes()))

        # Flag POIs
        for name, pos in self.goals.items():
            G.add_node(pos, name=name, type="goal")

        for name, pos in self.collections.items():
            G.add_node(pos, name=name, type="collection")

        for name, pos in self.initial_states.items():
            G.add_node(pos[:2], name=name, type="initial_state")

        # Connect Neighbors
        for i, u in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                dist = np.linalg.norm(np.array(u) - np.array(v))

                if dist <= connection_radius:
                    line = LineString([u, v])
                    if traversable_space.contains(line):
                        G.add_edge(u, v, weight=dist)

        if self.VERBOSE:
            print(f"Graph generated: {len(G.nodes)} nodes, {len(G.edges)} edges")
        self.G = G

    def get_graph(self) -> nx.Graph:
        return self.G

    def _calc_free_space(self, inflation_dist: float = 0.0) -> BaseGeometry:
        static_obstacles = self.obstacles[:-1]
        boundary_wrapper = self.obstacles[-1]

        # --- Ensure Boundary is a Polygon ---
        map_boundary_shape = boundary_wrapper.shape
        if isinstance(map_boundary_shape, LinearRing):
            # Convert "Wireframe" boundary to "Filled" Polygon
            map_boundary_poly = Polygon(map_boundary_shape)
        else:
            map_boundary_poly = map_boundary_shape

        # Inflate world boundaries inward
        navigable_boundary = map_boundary_poly.buffer(-inflation_dist, join_style="mitre")

        assert (
            not navigable_boundary.is_empty
        ), "    [ERROR]: Robot radius + margin too large. Boundary inflation left empty map."

        # Inflate obstacles outwards
        obs_shapes = [obs.shape for obs in static_obstacles]
        merged_obstacles = unary_union(obs_shapes)
        inflated_obstacles = merged_obstacles.buffer(inflation_dist, join_style="round", resolution=3)

        # Free Space = (Shrunk Boundary) - (Inflated Obstacles)
        traversable_space = navigable_boundary.difference(inflated_obstacles)

        assert (
            not traversable_space.is_empty
        ), "    [ERROR]: Robot radius + margin too large. Obstacle inflation left empty map."

        return traversable_space

    def _calc_grid_nodes(self) -> np.ndarray:

        # Physics loop settings
        relaxation_max_iter = 20
        relaxation_step_size = 0.1

        # --- Generate Hexagonal Grid ---
        xs = np.arange(self.min_x, self.max_x, self.node_dist, dtype=float)
        ys = np.arange(self.min_y, self.max_y, self.node_dist * np.sqrt(3) / 2.0, dtype=float)

        # Offset every other row to create hexes from squares
        candidates = []
        for i, y in enumerate(ys):
            offset = (self.node_dist / 2) if i % 2 == 1 else 0
            for x in xs:
                candidates.append([x + offset, y])

        valid_candidates = []
        free_space = self.free_space
        for cand in candidates:
            if free_space.contains(Point(cand)):
                valid_candidates.append(cand)

        # --- Add Goals & Collections to start of valid_candidates ---
        all_nodes = np.array(self.goal_pos + self.collections_pos + self.initial_state_pos + candidates)

        if GlobalConfig.RELAX_GRAPH:
            # Apply Force-Directed Relaxation
            final_nodes = self._relax_nodes(
                initial_nodes=all_nodes,
                relaxation_max_iter=relaxation_max_iter,
                relaxation_step_size=relaxation_step_size,
            )
        valid_polys = self._calc_free_space(self.inflation_dist * 1.2)
        final_nodes = self._move_points_to_legal(all_nodes, valid_polys)

        return final_nodes

    def _move_points_to_legal(self, positions: np.ndarray, valid_area: BaseGeometry) -> np.ndarray:
        for idx in range(self.num_pois, len(positions)):
            pt = Point(positions[idx])
            if not valid_area.contains(pt):
                nearest_p, _ = nearest_points(valid_area, pt)
                positions[idx] = np.array([nearest_p.x, nearest_p.y])
        return positions

    def _relax_nodes(
        self,
        initial_nodes: np.ndarray,
        relaxation_max_iter: int,
        relaxation_step_size: float,
    ) -> np.ndarray:
        """
        Iteratively moves nodes based on forces and constraints.
        """

        def _interaction_force(dist: float, target_dist: float) -> float:
            if target_dist - dist > 0.0:
                k = 2.0
            else:
                k = 0.1
            force = k * (target_dist - dist)

            return force

        current_positions = initial_nodes.copy()
        last_positions = current_positions.copy()
        n_nodes = len(current_positions)

        if self.VERBOSE:
            print(f"Relaxing {n_nodes} nodes over {relaxation_max_iter} iterations...")

        valid_polys = self._calc_free_space(self.inflation_dist * 1.2)

        last_max_change = np.inf

        for it in range(relaxation_max_iter):
            # Move points in raw boundary to nearest traversible space, but skip POI
            current_positions = self._move_points_to_legal(current_positions, valid_polys)

            if GlobalConfig.PLOT_RELAXATION:

                self.G = nx.Graph()
                for node in current_positions:
                    node = (float(node[0]), float(node[1]))
                    self.G.add_node(node, pos=node)

                # Flag POIs
                for name, pos in self.goals.items():
                    self.G.add_node(pos, name=name, type="goal")

                for name, pos in self.collections.items():
                    self.G.add_node(pos, name=name, type="collection")

                for name, pos in self.initial_states.items():
                    self.G.add_node(pos[:2], name=name, type="initial_state")

                self.plot(filename=f"relax_it{it}.png")

            # Build KDTree for efficient neighbor lookup
            tree = KDTree(current_positions)

            # Find neighbors within interaction range
            interaction_radius = self.node_dist * 2.0
            pairs = tree.query_pairs(r=interaction_radius)

            displacements = np.zeros_like(current_positions)

            # Calculate Forces
            for i, j in pairs:
                p_i = current_positions[i]
                p_j = current_positions[j]

                vec_ij = p_i - p_j
                dist = float(np.linalg.norm(vec_ij))

                # Get scalar force magnitude
                f_mag = _interaction_force(dist, self.node_dist)

                # Force vector direction (i pushed by j)
                if dist > 1e-6:
                    f_vec = (vec_ij / dist) * f_mag
                else:
                    f_vec = np.random.rand(2)  # Random kick if on top of each other

                displacements[i] += f_vec
                displacements[j] -= f_vec

            # --- Zero out displacements for goals and collections ---
            displacements[: self.num_pois] = 0.0

            # Apply Update (with damper)
            current_positions += displacements * relaxation_step_size

            calc_diff = current_positions - last_positions
            max_change = np.max(np.abs(calc_diff)) if len(calc_diff) > 0 else 0

            last_positions = current_positions.copy()
            threshold = self.node_dist / 20
            if self.VERBOSE:
                print(f"   max change: {max_change:.3f} ({threshold:.3f} allowed)")
            if max_change < threshold or max_change >= last_max_change:
                if self.VERBOSE:
                    print(f"Relaxation converged. Final change = {max_change:.3f} (n_nodes = {len(current_positions)})")
                break
            last_max_change = max_change
            if it == 19 and self.VERBOSE:
                print(
                    f"Relaxation stoped before converging. Final change = {max_change:.3f} (n_nodes = {len(current_positions)})"
                )

        return current_positions

    def find_path_in_graph(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        """
        start and goal node have to be a part of the graph already
        """

        query_G = self.G.copy()

        query_G.add_node("start", pos=start)
        query_G.add_node("goal", pos=goal)

        node_ids = list(self.G.nodes)
        node_coords = np.array([self.G.nodes[n]["pos"] for n in node_ids])

        # Build KDTree for fast lookup
        tree = KDTree(node_coords)
        k_neighbors = 15  # Connect to 15 nearest nodes

        for name, pos in [("start", start), ("goal", goal)]:
            dists, indices = tree.query(pos, k=k_neighbors)

            connections = 0
            for d, idx in zip(dists, indices):
                target_node_id = node_ids[idx]
                target_pos = node_coords[idx]

                line = LineString([pos, target_pos])
                if self.traversable_space.contains(line):
                    query_G.add_edge(name, target_node_id, weight=d)
                    connections += 1

            if connections == 0 and self.VERBOSE:
                print(f"  [Path] Warning: Could not connect '{name}' to any graph nodes!")

        def heuristic(u, v) -> float:
            """
            heuristic function for A* in find_path_in_graph function
            """
            return float(np.linalg.norm(np.subtract(u, v), 0))

        try:
            path_nodes = nx.astar_path(
                self.G,
                source="start",
                target="goal",
                heuristic=heuristic,
                weight="weight",
            )
            path_coords = []
            for n in path_nodes:
                if n == "start":
                    path_coords.append(start)
                elif n == "goal":
                    path_coords.append(goal)
                else:
                    path_coords.append(n)
            if self.VERBOSE:
                print("     Path found")
            return path_coords
        except nx.NetworkXNoPath:
            if self.VERBOSE:
                print("     No path found.")
            return []

    def plot(
        self,
        filename: Optional[str] = None,
        sub_dir: Optional[str] = None,
        path: Optional[list[tuple[float, float]]] = None,
    ):
        plot_graph(
            self.G,
            filename=filename,
            sub_dir=sub_dir,
            planned_path=path,
            obstacles=self.obstacles,
            traversable_space=self.traversable_space,
        )
        return

    def _calc_random_nodes(self, n_samples: int) -> List[Point]:
        if self.traversable_space.is_empty:
            print("   [ERROR] Free space is EMPTY! Cannot generate graph.")
            return

        max_attempts = n_samples * 10
        rng = np.random.default_rng(seed=42)
        count = 0
        attempts = 0
        min_x, min_y, max_x, max_y = self.min_x, self.min_y, self.max_x, self.max_y
        traversable_space = self.traversable_space
        G = self.G

        while count < n_samples and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            p = Point(x, y)

            if traversable_space.contains(p):
                G.add_node((x, y), pos=(x, y))
                count += 1

        nodes = list(G.nodes)
        return nodes

    def get_traversable_space(self) -> BaseGeometry:
        return self.traversable_space
