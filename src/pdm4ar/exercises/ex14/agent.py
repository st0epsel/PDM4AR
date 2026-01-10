from functools import reduce
from gc import is_finalized
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from os import close
from re import S
from typing import Mapping, Optional, List, Tuple, Sequence, Dict
import math

import numpy as np
import networkx as nx


from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree, cKDTree
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import from_wkt
from shapely.ops import unary_union, nearest_points

from pdm4ar.exercises.ex14.global_config import GlobalConfig
from pdm4ar.exercises.ex14.graph import GraphBuilder
from pdm4ar.exercises.ex14.helper_fcts import plot_graph
from pdm4ar.exercises.ex14.trajectory_tracker import TrajectoryTracker
from pdm4ar.exercises.ex14.agent_state import AgentState
from pdm4ar.exercises.ex14.global_planner import GlobalPlanMessage, Pdm4arGlobalPlanner


@dataclass(frozen=True)
class Pdm4arAgentParams:
    coll_dist_1: float = 0.8  # [s]
    coll_dist_2: float = 2.0  # [s]
    # replaning_angle: tuple[float, float] = (np.pi / 4, -np.pi / 4)
    min_col_radius: float = GlobalConfig.MIN_COL_RADIUS  # [m]
    abort_retreat: float = GlobalConfig.ABORT_RETREAT


class Pdm4arAgent(Agent):
    def __init__(self):
        self.name: PlayerName
        self.G_originaloal: PlanningGoal
        self.static_obstacles: Sequence[StaticObstacle]
        self.sg: DiffDriveGeometry
        self.sp: DiffDriveParameters

        self.params = Pdm4arAgentParams()

        # Debugging
        self.VERBOSE = GlobalConfig.AGENT_VERBOSE
        self.PLOT = GlobalConfig.PLOT_AGENT_PLANS
        self.last_print_time = 0.0

        # Goal tracking
        self.G_original: nx.Graph
        self.G_original_nodes: List
        self.kd_tree_original: cKDTree
        self.G_: nx.Graph
        self.G_nodes: List
        self.kd_tree: cKDTree
        self.waypointslist: np.ndarray
        self.waypoint_counter: int = 0
        self.target: np.ndarray
        self.coinID: Optional[str] = None
        self.last_coinID: Optional[str] = None
        self.traversable_space: BaseGeometry
        self.traversable_space_original: BaseGeometry

        # Obstacle avoidance and tracking
        self.priority: float = 0
        self.priority_map: Dict[str, float] = {}
        self.state: AgentState
        self.other_agents: Dict[str, AgentState] = {}

        self.avoid_coins: list[tuple[float, float]] = []
        self.current_target: Optional[tuple[float, float]] = None
        self.dist_to_target = np.inf
        self.was_booped = False
        self.local_max_omega: float = 0.0
        self.reverse_counter: int = 0
        self.retreat_node: Optional[tuple[float, float]] = None

        # Robot Parameters
        self.omega_max: Tuple[float, float] = (-10.0, 10.0)
        self.wheel_radius: float = 0.0
        self.wheelbase: float = 0.0
        self.robot_radius: float = 0.0

        # Robot Driving
        self.planned_trajectory: TrajectoryTracker
        self.driving_direction = 1
        self.is_finished = False

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.name = init_sim_obs.my_name
        self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params

        # Robot Parameters
        self.omega_max = self.sp.omega_limits
        self.wheel_radius = self.sg.wheelradius
        self.wheelbase = self.sg.wheelbase
        self.robot_radius = self.sg.radius

        # --- NEW: Update AgentState Radius ---
        self.state = AgentState(self.robot_radius)

    def on_receive_global_plan(self, serialized_msg: str):

        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)

        initial_state_vec = global_plan.initial_states[self.name]

        self.state.update(np.array(initial_state_vec), sim_time=0.0)

        self.traversable_space_original = global_plan.traversable_space

        self.priority = global_plan.agent_workload[self.name]
        self.priority_map = global_plan.agent_workload
        waypointslist = global_plan.agents_goals[self.name]
        waypointslist.append(initial_state_vec[:2])
        self.waypointslist = np.array(waypointslist)

        if self.VERBOSE:
            print(f"   {self.name} waypointslist: \n{self.waypointslist.round(2)}")

        # calculate which coins to avoid (coins for other agents)
        collections = global_plan.collections
        for name, pois in global_plan.agents_goals.items():
            if name == self.name:
                continue
            for pos in pois:
                if pos not in collections:
                    self.avoid_coins.append(pos)

        # Delete Nodes -> CHANGE TO DELETE NODES PASSING THROUGH area.
        self.G_original = nx.node_link_graph(global_plan.graph_data)
        self.G_original_nodes = list(self.G_original.nodes)
        self.kd_tree_original = KDTree(self.G_original_nodes)
        for pos in self.avoid_coins:
            obs = Point(pos[0], pos[1]).buffer(self.state.radius + GlobalConfig.CRASH_BUFFER, resolution=3)
            self.traversable_space_original = self.traversable_space_original.difference(obs)
            nodes_indices = self.kd_tree_original.query_ball_point(
                pos, r=self.state.radius * (1 + GlobalConfig.CRASH_BUFFER)
            )
            nodes_to_remove = [self.G_original_nodes[i] for i in nodes_indices]
            self.G_original.remove_nodes_from(nodes_to_remove)
            self.G_original_nodes = list(self.G_original.nodes)
            self.kd_tree_original = KDTree(self.G_original_nodes)

        self.traversable_space = self.traversable_space_original

        self.G = self.G_original
        self.G_nodes = self.G_original_nodes
        self.kd_tree = self.kd_tree_original

        self.target = self.waypointslist[self.waypoint_counter]

        # Use self.state.x / self.state.y
        if self.VERBOSE:
            print(f"   self.state: {self.state()}")
            print(f"   target: {self.target}")
        trajectory = self.calc_shortest_path(start=(self.state.x, self.state.y), goal=tuple(self.target))
        if trajectory is None or len(trajectory) <= 0:
            trajectory = [tuple(self.target)]
        self.planned_trajectory = TrajectoryTracker(trajectory=trajectory)

        if self.PLOT:
            plot_graph(G=self.G_original, filename=f"agent_graph_{self.name}", planned_path=trajectory)

        self.local_max_omega = np.max(self.omega_max)

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds."""

        self.sim_time = float(sim_obs.time)

        # --- Update own state ---
        raw_state = sim_obs.players[self.name].state
        state_vec = np.array([float(raw_state.x), float(raw_state.y), float(raw_state.psi)])
        self.dist_to_target = math.hypot(self.state.x - self.target[0], self.state.y - self.target[1])
        if self.reverse_counter <= 0:
            self.planned_trajectory.log_point((state_vec[0], state_vec[1]))

        # --- Update own coin state ---
        self.coinID = sim_obs.players[self.name].collected_goal_id
        coin_changed = self.coinID != self.last_coinID
        self.last_coinID = self.coinID
        my_base = self.priority_map.get(self.name, 0.0)
        self.priority = my_base + 99.0 if self.coinID is not None else my_base
        self.state.update(state_vec, self.sim_time, self.priority)

        if coin_changed and self.VERBOSE:
            print(f"self.name: {self.name} - coinID: {self.coinID}")

        # --- Update Other observable Agents ---
        observed_agent_names = set()
        for p_name, p_obs in sim_obs.players.items():
            if p_name == self.name:
                continue

            observed_agent_names.add(p_name)
            # Create if not exists
            if p_name not in self.other_agents:
                self.other_agents[p_name] = AgentState(radius=self.robot_radius)
            # Update their state
            p_raw = p_obs.state
            p_vec = np.array([float(p_raw.x), float(p_raw.y), float(p_raw.psi)])

            other_base = self.priority_map.get(p_name, 0.0)
            other_has_coin = sim_obs.players[p_name].collected_goal_id is not None
            other_prio = other_base + 99.0 if other_has_coin else other_base
            self.other_agents[p_name].update(p_vec, self.sim_time, other_prio)

        # Delete if not observed anymore
        for stored_name in list(self.other_agents.keys()):
            if stored_name not in observed_agent_names:
                replanning = True
                del self.other_agents[stored_name]

        # --- Collision detection ---
        min_col_dist = np.inf
        colliding_agents: dict[str, float] = {}
        closest_colliding_agent = None
        for other_name, other in self.other_agents.items():
            dist = self.state.get_distance(other) - 2 * self.state.radius
            if dist < 2.0 and dist < min_col_dist:
                colliding_agents[other_name] = dist
                min_col_dist = dist
                closest_colliding_agent = other_name

        if self.VERBOSE and closest_colliding_agent is not None:
            dist = colliding_agents[closest_colliding_agent]
            """print(f"self.name: {self.name}")
            print(
                f"   Robot {closest_colliding_agent} is {dist:.2f} m away from a collision with {self.name} at time {self.sim_time}"
            )"""

        # --- Update Target Logic ---
        replanning = False
        if (coin_changed or self.dist_to_target < 0.3) and not self.is_finished:
            self.waypoint_counter += 1
            if self.VERBOSE:
                print(f"   Updated Target (counter: {self.waypoint_counter})")
            if self.waypoint_counter >= len(self.waypointslist):
                self.is_finished = True
            else:
                self.target = self.waypointslist[self.waypoint_counter]
                replanning = True

        # --- Collision Avoidance ---
        avoidance_action = "continue"  # one of "continue", "break", "reverse" or "replan" or "retreat"
        # replanning leads to either continue for a good path or "reverse" for a bad path

        if closest_colliding_agent is not None:
            other_prio = self.other_agents[closest_colliding_agent].priority
            self_higher_prio = other_prio < self.state.priority and not self.is_finished
            col_dist = colliding_agents[closest_colliding_agent]
            other_psi = self.other_agents[closest_colliding_agent].psi

            if self_higher_prio:
                if self.state.is_in_front_of(self.other_agents[closest_colliding_agent]):
                    avoidance_action = "continue"
                if col_dist < self.params.min_col_radius:
                    avoidance_action = "replan"
                else:
                    avoidance_action = "continue"
            else:
                if self.state.is_in_front_of(self.other_agents[closest_colliding_agent]):
                    avoidance_action = "continue"
                if col_dist < self.params.min_col_radius:
                    avoidance_action = "retreat"
                else:
                    avoidance_action = "continue"
            if self.VERBOSE and avoidance_action != "continue":
                print(f"{self.name} - time: {self.sim_time}")
                print(
                    f"   Robot {closest_colliding_agent} is {dist:.2f} m away from a collision with {self.name} at time {self.sim_time}"
                )
                print(f"   Recommended action: {avoidance_action}")

        # --- Local Graph Prep ---
        # Calculate local Graph and local KDTree
        # Calculate local traversable_space
        pos_to_avoid = []
        for name in colliding_agents.keys():
            pos_to_avoid.append(
                (
                    self.other_agents[name].x,
                    self.other_agents[name].y,
                )
            )
        if closest_colliding_agent is None:
            self.G = self.G_original
            self.kd_tree = self.kd_tree_original
            self.traversable_space = self.traversable_space_original
        else:
            if self.state.priority > self.other_agents[closest_colliding_agent].priority:
                infl_radius = 2 * (1 + GlobalConfig.CRASH_BUFFER) * self.state.radius
            else:
                infl_radius = GlobalConfig.NOT_YIELDING_AGNET_BUFFER * self.state.radius
            # print(f"   pos to avoid: {pos_to_avoid} (radius: {infl_radius})")
            self.G = self.G_original.copy()
            self.G_nodes = list(self.G.nodes)
            self.kd_tree = KDTree(self.G_nodes)
            self.traversable_space = self.traversable_space_original
            for pos in pos_to_avoid:
                obs = Point(pos[0], pos[1]).buffer(infl_radius, resolution=3)
                self.traversable_space = self.traversable_space.difference(obs)
                nodes_indices = self.kd_tree.query_ball_point(pos, r=infl_radius)
                nodes_to_remove = [self.G_nodes[i] for i in nodes_indices]
                self.G.remove_nodes_from(nodes_to_remove)
                self.G_nodes = list(self.G.nodes)
                self.kd_tree = KDTree(self.G_nodes)

        # --- Apply evasive action ---
        if avoidance_action == "continue":
            self.local_max_omega = min(np.max(self.omega_max), self.local_max_omega * 1.6)
            if self.local_max_omega < max(self.omega_max):
                print(f"   normal driving (omega_max: {self.local_max_omega})")
        elif avoidance_action == "retreat":
            self.retreat_node = self.get_max_angle_retreat_node(
                self.other_agents[closest_colliding_agent], desired_dist=GlobalConfig.RETREAT_DIST
            )
            if self.VERBOSE:
                print(f"   Evading. retreat node: {self.retreat_node}")
            if self.retreat_node is not None:
                new_trajectory = self.calc_shortest_path((self.state.x, self.state.y), self.retreat_node)
                if new_trajectory is not None:
                    self.planned_trajectory.new_traj(new_trajectory)
                else:
                    print(f"FINDING EVASION TRAJECTORY FAILED")
            else:
                print(f"FINDING EVASION NODE FAILED")
        elif avoidance_action == "replan":
            replanning = True

        else:
            assert False, "Correct writing would be great you know - Mistaken avoidance action spelling"

        # --- Replanning ---
        # If not evading and target not at the end of the trajectory
        if np.all(self.target != self.planned_trajectory.trajectory[-1]) and self.retreat_node is None:
            replanning = True

        if replanning:
            new_trajectory = self.replan(tuple(self.target))
            if new_trajectory is not None and len(new_trajectory) >= 2:
                self.planned_trajectory.new_traj(new_trajectory)
            else:
                return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        # When retreat node is reached, return to normal business
        if (
            self.retreat_node is not None
            and self.planned_trajectory.traj_finished
            or closest_colliding_agent is None
            or closest_colliding_agent is not None
            and colliding_agents[closest_colliding_agent] > self.params.abort_retreat
        ):
            self.retreat_node = None

        # --- Generate Control Outputs ---
        if self.is_finished and avoidance_action == "continue":
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        if self.reverse_counter <= 0:
            tx, ty, k_curvature = self.planned_trajectory.get_lookahead_point_and_curvature(self.state.x, self.state.y)
        else:
            if self.VERBOSE:
                print(f"   reverse_counter: {self.reverse_counter}")
            self.local_max_omega = max(self.omega_max)
            tx, ty, k_curvature = self.planned_trajectory.reverse_point_and_curvature(self.state.x, self.state.y)
            self.reverse_counter -= 1

        ex = tx - self.state.x
        ey = ty - self.state.y
        self.tracking_error = np.linalg.norm([ex, ey])

        desired_heading = np.arctan2(ey, ex)
        alpha = desired_heading - self.state.psi
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        direction = 1
        if abs(alpha) > np.pi / 2:
            alpha = alpha - math.copysign(math.pi, alpha)
            direction = -1

        wl, wr = self.curvature_control(
            heading_error=alpha, curvature=k_curvature, max_wheel_speed=self.local_max_omega, direction=direction
        )

        wr = np.clip(wr, self.omega_max[0], self.omega_max[1])
        wl = np.clip(wl, self.omega_max[0], self.omega_max[1])

        # Plotting Logic
        if self.retreat_node is not None:
            out_msg = f"({self.retreat_node[0]:.2f}, {self.retreat_node[1]:.2f})"
        else:
            out_msg = f"{avoidance_action}"
        if self.PLOT and float(self.last_print_time) + 1.0 <= self.sim_time:
            plot_graph(
                filename=f"{self.name}_time_{self.sim_time}_{out_msg}",
                G=self.G,
                planned_path=self.planned_trajectory.get_trajectory(),
                actual_path=self.planned_trajectory.get_actual_traj(),
                other_agents=self.other_agents,
                own_state=self.state,
                traversable_space=self.traversable_space,
            )
            self.last_print_time = self.sim_time

        return DiffDriveCommands(omega_l=wl, omega_r=wr)

    def get_max_angle_retreat_node(
        self, other: "AgentState", desired_dist: float = 2.0
    ) -> Optional[tuple[float, float]]:
        # 1. Calculate the ideal "Run Away" vector (Other -> Self)
        ideal_vx = self.state.x - other.x
        ideal_vy = self.state.y - other.y
        ideal_norm = math.hypot(ideal_vx, ideal_vy)

        # 2. Find candidates within a ring
        min_r = desired_dist * 0.75
        max_r = desired_dist * 1.25

        # Get all indices within max radius
        candidate_indices = self.kd_tree.query_ball_point([self.state.x, self.state.y], r=max_r)

        best_node = None
        max_score = -2.0  # Cosine similarity ranges from -1 to 1

        for i in candidate_indices:
            node_pos = self.G_nodes[i]

            # Vector from Self -> Node
            node_vx = node_pos[0] - self.state.x
            node_vy = node_pos[1] - self.state.y
            dist = math.hypot(node_vx, node_vy)

            # Filter out nodes that are too close
            if dist < min_r:
                continue

            # 3. Calculate Score (Cosine Similarity)
            dot_product = (ideal_vx * node_vx) + (ideal_vy * node_vy)
            score = dot_product / (ideal_norm * dist)

            if score > max_score:
                max_score = score
                best_node = node_pos

        return best_node

    def replan(self, goal: tuple[float, float]) -> Optional[list[tuple[float, float]]]:
        new_trajectory = self.calc_shortest_path(start=(self.state.x, self.state.y), goal=goal)

        if self.VERBOSE:
            print(f"   Replanning...")
            print(f"   sucessfull: {self.planned_trajectory.check_traj(new_trajectory)}")

        if self.PLOT:
            plot_graph(
                filename=f"{self.name}_time_{self.sim_time}_REPLANNING",
                G=self.G,
                actual_path=self.planned_trajectory.get_actual_traj(),
                planned_path=new_trajectory,
                other_agents=self.other_agents,
                own_state=self.state,
                traversable_space=self.traversable_space,
            )
        """print(f"   new trajectory:")
        print(f"   {new_trajectory}")"""

        if self.planned_trajectory.check_traj(new_trajectory):
            return new_trajectory
        else:
            return None

    def calc_shortest_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        smooth: bool = GlobalConfig.SMOOTH_PATH,
        reduce_nodes: bool = GlobalConfig.REDUCE_PATH_NODES,
    ) -> Optional[list[tuple[float, float]]]:
        def heuristic(u, v) -> float:
            return float(np.linalg.norm(np.subtract(u, v), 0))

        start_node = self.get_nearest_node(start)
        goal_node = goal

        # Safety check: If we pruned the graph, start/goal nodes might be gone.
        if goal_node not in self.G:
            if self.VERBOSE:
                print("  [Plan] Goal node is in obstacle zone.")
            return None

        try:
            path_coords = nx.astar_path(
                self.G,
                source=start_node,
                target=goal_node,
                heuristic=heuristic,
                weight="weight",
            )
            if reduce_nodes:
                path_coords = self.reduce_nodes(path_coords)

            if smooth:
                path_coords = self.smooth_path(path=path_coords, smoothness=0.0)
            return path_coords

        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            if self.VERBOSE:
                print(f"    No path found: {e}")
            return None

    def calc_path_length(self, path: List[Tuple[float, float]]) -> float:
        length = 0.0
        for i in range(1, len(path)):
            length += np.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        return length

    def get_nearest_node(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        self.G_nodes = list(self.G.nodes)
        self.kd_tree = KDTree(self.G_nodes)
        dist, idx = self.kd_tree.query(pos)
        if dist > 4 * self.state.radius:
            print(f"NEAREST NODE IS A LIL FAR AWAY, INIT???")
        return self.G_nodes[idx]

    def smooth_path(self, path: List[Tuple[float, float]], smoothness=0.0) -> List[Tuple[float, float]]:
        """
        Interpolates the jagged grid path into a smooth B-Spline.
        """
        k = 3
        if len(path) <= 1:
            return path
        elif len(path) <= 4:
            dist = self.calc_path_length(path)
            num_points = int(dist / 0.05)
            res = np.linspace(0, 1, num_points)
            points = np.array(path[0]) + res[:, np.newaxis] * (np.array(path[1]) - np.array(path[0]))
            return [tuple(point) for point in points]

        clean_path = path

        x = [p[0] for p in clean_path]
        y = [p[1] for p in clean_path]

        try:
            # s=smoothing factor.
            tck, u = splprep([x, y], s=smoothness, k=k)

            # Generate new points
            # Generate roughly one point every 5cm
            dist = self.calc_path_length(clean_path)
            num_points = int(dist / 0.05)

            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)

            return list(zip(x_new, y_new))

        except Exception as e:
            if self.VERBOSE:
                print(f"[Smooth] Spline failed: {e}, returning raw path")
            return path

    def curvature_control(
        self,
        heading_error: float,
        curvature: float,
        max_wheel_speed: float,
        direction: int = 1,
        state_error_gain: float = GlobalConfig.STATE_ERROR_GAIN,
        ff_gain: float = GlobalConfig.FF_GAIN,
    ) -> tuple[float, float]:
        """
        Generates wheel commands using pure pursuit curvature + P-control.
        Correctly handles the inversion of steering logic when reversing.
        """

        # 1. Calculate Desired Geometric Curvature
        # FEEDFORWARD: 'curvature' is the geometric shape of the path.
        #   It stays the same regardless of direction.
        # FEEDBACK: 'heading_error' needs to be inverted when reversing.
        #   Forward: Error < 0 (Right) -> Steer Right (Neg k) -> Turn CW.
        #   Backward: Error < 0 (Right) -> Steer Left (Pos k) -> Turn CW.
        k_des = ff_gain * curvature + (state_error_gain * heading_error * direction)

        # 2. Wheel Speed Calculation (Time-Optimal)
        # Keep the "Outer" wheel at max speed to turn as fast as possible.
        L = self.wheelbase
        target_speed = max_wheel_speed * direction

        if k_des > 0:  # Geometric LEFT Turn
            # For a Left turn, the Right wheel travels the longer arc (Outer).
            wr = target_speed

            # Inner wheel (Left)
            # Formula: vl = vr * (2 - kL)/(2 + kL)
            numerator = 2 - (k_des * L)
            denominator = 2 + (k_des * L)

            if abs(denominator) < 1e-6:
                wl = -target_speed
            else:
                wl = wr * numerator / denominator

        else:  # Geometric RIGHT Turn
            # For a Right turn, the Left wheel travels the longer arc (Outer).
            wl = target_speed

            # Inner wheel (Right)
            # Formula: vr = vl * (2 + kL)/(2 - kL)
            numerator = 2 + (k_des * L)
            denominator = 2 - (k_des * L)

            if abs(denominator) < 1e-6:
                wr = -target_speed
            else:
                wr = wl * numerator / denominator

        return wl, wr

    def reduce_nodes(self, trajectory: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        length = len(trajectory)
        if length < 3:
            return trajectory

        reduced_trajectory = [trajectory[0]]
        current_idx = 0
        MAX_DIST = 0.8  # Maximum allowed distance between consecutive points

        # Keep going until we reach the last node
        while current_idx < length - 1:
            found_shortcut = False

            # Iterate backwards from the End of the path down to the current node
            for next_idx in range(length - 1, current_idx, -1):

                start_p = trajectory[current_idx]
                end_p = trajectory[next_idx]

                # Check validity:
                # 1. Immediate neighbor is always valid
                # 2. Distant node is valid if LineString is contained in free space
                is_valid = next_idx == current_idx + 1
                if not is_valid:
                    line = LineString([start_p, end_p])
                    if self.traversable_space.contains(line):
                        is_valid = True

                if is_valid:
                    # --- INTERPOLATION LOGIC ---
                    # Calculate distance for the valid segment
                    # Cast to numpy array ensures subtraction works if points are tuples
                    dist = np.linalg.norm(np.array(end_p) - np.array(start_p))

                    if dist > MAX_DIST:
                        # Calculate how many segments we need
                        num_segments = int(np.ceil(dist / MAX_DIST))

                        # Generate intermediate points
                        # num_segments + 1 ensures we get the endpoints correctly
                        x_vals = np.linspace(start_p[0], end_p[0], num_segments + 1)
                        y_vals = np.linspace(start_p[1], end_p[1], num_segments + 1)

                        # Append interpolated points
                        # Start from index 1 because index 0 is 'start_p' (already in list)
                        for i in range(1, len(x_vals)):
                            reduced_trajectory.append((x_vals[i], y_vals[i]))
                    else:
                        # Segment is short enough, just add the end point
                        reduced_trajectory.append(end_p)

                    # Update indices
                    current_idx = next_idx
                    found_shortcut = True
                    break

            # Safety catch
            if not found_shortcut:
                current_idx += 1
                # Even in safety fallback, we technically might want to check dist,
                # but usually graph nodes are close enough.
                reduced_trajectory.append(trajectory[current_idx])

        return reduced_trajectory
