from typing import List, Tuple, Dict, Mapping

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


class GlobalPlanMessage(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    agents_goals: Dict[str, List[Tuple[float, float]]]  # Dict contains name: List of POI positions
    agent_workload: Dict[str, float]
    initial_states: Dict[str, Tuple[float, float, float]]
    graph_data: Dict
    collections: List[tuple[float, float]]
    traversable_space: BaseGeometry

    # --- SERIALIZER (Automatic when dumping to JSON) ---
    @field_serializer("traversable_space")
    def serialize_geometry(self, value: BaseGeometry, _info) -> str:
        return value.wkt

    # --- VALIDATOR/DESERIALIZER (Automatic when loading from JSON) ---
    @field_validator("traversable_space", mode="before")
    @classmethod
    def deserialize_geometry(cls, value) -> BaseGeometry:
        if isinstance(value, str):
            return from_wkt(value)
        return value


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        self.sg = DiffDriveGeometry.default()
        self.G_original = None
        self.dict_of_initial_states: Dict[str, Tuple[float, float, float]] = {}
        self.dict_of_collections: Dict[str, Tuple[float, float]] = {}
        self.dict_of_goals: Dict[str, Tuple[float, float]] = {}
        self.G_originalg_obal_graph: GraphBuilder
        self.VERBOSE = GlobalConfig.GLOBAL_PLANNER_VERBOSE

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # TODO: implement here your global planning stack.
        # init_sim_obs.shared_goals = dict with the goal names and locations
        # init_sim_obs.collection_points = dict with the dropoff names and locations
        # init_sim_obs.dg_scenario = dg_commons.sim.scenarios.structures.DgScenario ???
        # init_sim_obs.initial_states = dict with player names and starting pos
        # init_sim_obs.players_obs = dict with player names and InitSimObs object with name, parameters, geometry etc
        # init_sim_obs.seed = int

        static_obs = init_sim_obs.dg_scenario.static_obstacles

        dict_of_initial_states = {}
        for player_name, state in init_sim_obs.initial_states.items():
            dict_of_initial_states[player_name] = (state.x, state.y, state.psi)
        self.dict_of_initial_states = dict_of_initial_states

        dict_of_goals = {}
        list_of_coins = []
        for goal_id, goal_data in init_sim_obs.shared_goals.items():
            center_x = goal_data.polygon.centroid.x
            center_y = goal_data.polygon.centroid.y
            dict_of_goals[goal_id] = (center_x, center_y)
            list_of_coins.append(
                (
                    center_x,
                    center_y,
                )
            )
        self.dict_of_goals = dict_of_goals

        dict_of_collections = {}
        list_of_collections = []
        for collection_id, collection_data in init_sim_obs.collection_points.items():
            center_x = collection_data.polygon.centroid.x
            center_y = collection_data.polygon.centroid.y
            dict_of_collections[collection_id] = (center_x, center_y)
            list_of_collections.append((center_x, center_y))
        self.dict_of_collections = dict_of_collections

        if self.VERBOSE:
            print("   Building Graph...")
        self.G_originalg_obal_graph = GraphBuilder(
            static_obs,
            robot_radius=0.6,
            safety_margin=GlobalConfig.CRASH_BUFFER,
            n_samples=GlobalConfig.N_NODES,  # Tunable: More samples = better paths but slower init
            node_connection_mult=GlobalConfig.NODE_CONNECTION_MULT,
            goals=dict_of_goals,
            collections=dict_of_collections,
            initial_sates=dict_of_initial_states,
        )

        filename = f"{GlobalConfig.SESSION_NAME}_{np.random.random_sample():.3f}.png"
        self.G_originalg_obal_graph.plot(filename=filename)

        serialized_graph_data = nx.node_link_data(self.G_originalg_obal_graph.get_graph())
        if self.VERBOSE:
            print(f"   Assigning Goals...")
        agents_goals, agent_workload = self.assign_goals(init_sim_obs=init_sim_obs)
        if self.VERBOSE:
            print(f"   Creating Message...")
        global_plan_message = GlobalPlanMessage(
            agents_goals=agents_goals,
            agent_workload=agent_workload,
            initial_states=dict_of_initial_states,
            graph_data=serialized_graph_data,
            collections=list_of_collections,
            traversable_space=self.G_originalg_obal_graph.get_traversable_space(),
        )

        return global_plan_message.model_dump_json(round_trip=True)

    def assign_goals(self, init_sim_obs) -> Tuple[Dict[str, List], Dict[str, float]]:

        unassigned_goals = self.dict_of_goals
        collection_map = self.dict_of_collections

        def dist_heuristic(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        def get_path_cost(u: Tuple[float, float], v: Tuple[float, float]) -> float:
            try:
                return nx.astar_path_length(G, u, v, heuristic=dist_heuristic, weight="weight")
            except nx.NetworkXNoPath as e:
                print(f"   nx.NetworkXNoPath Error - {e}")
                return np.inf

        def get_nearest_node(pos: Tuple[float, float], k: int = 1) -> Tuple[float, float]:
            dist, idx = kdtree.query(pos, k=k)
            if k > 1:
                idx = idx[-1]
            return tuple(node_list[idx])

        G = self.G_originalg_obal_graph.get_graph()
        node_list = list(G.nodes)
        kdtree = KDTree(node_list)

        # Agent initialization
        agent_states = {}
        for name, state in init_sim_obs.initial_states.items():
            start_pos = (state.x, state.y)
            start_node = get_nearest_node(start_pos)

            agent_states[name] = {
                "start_node": start_node,  # Current logical position on graph
                "workload": 0.0,  # Total distance assigned
                "plan": [],  # List of POIs
            }

        if GlobalConfig.GLOBAL_PLANNER_VERBOSE:
            print(
                f"\n[GlobalPlanner] Starting allocation for {len(unassigned_goals)} coins and {len(agent_states)} agents."
            )

        curr_node = (0, 0)
        k_max = int(GlobalConfig.LOOK_CLOSE_TO_UNREACHABLE_NODES * 1 + 1)

        while unassigned_goals:

            # 1. Pick the agent with the lowest workload
            curr_agent_name = min(agent_states, key=lambda k: agent_states[k]["workload"])
            curr_agent = agent_states[curr_agent_name]
            curr_node: Tuple[float, float] = curr_agent["start_node"]

            if self.VERBOSE:
                print(f"current agent: {curr_agent}")

            # 2. Find the best Coin -> Dropoff pair for this agent
            best_goal_id = None
            best_cost = float("inf")
            best_goal_node = None
            best_drop_node = None

            for g_id, g_pos in unassigned_goals.items():
                k: int = (
                    1  # If node is not reachable (inside an obstacle), at least try to get close -> likely gets picked up anyways
                )
                dist_to_coin = float("inf")
                while dist_to_coin == float("inf") and k <= k_max:
                    g_pos_approx = get_nearest_node(g_pos, k=k)
                    dist_to_coin = get_path_cost(get_nearest_node(curr_node, k=k), g_pos_approx)
                    k += 1

                if self.VERBOSE:
                    print(f"   path to g_ID: {g_id} - dist: {dist_to_coin}")

                if dist_to_coin == float("inf"):
                    continue

                # Find nearest dropoff for this specific coin
                nearest_drop_dist = float("inf")
                nearest_drop_node = None

                for did, d_pos in collection_map.items():

                    # Quick Euclidean check optimization
                    if dist_heuristic(g_pos_approx, d_pos) < nearest_drop_dist:
                        # Detailed A* check
                        k: int = (
                            1  # If node is not reachable (inside an obstacle), at least try to get close -> likely gets picked up anyways
                        )
                        d = float("inf")
                        while d == float("inf") and k <= 2:
                            d_pos_approx = get_nearest_node(d_pos, k=k)
                            d = get_path_cost(g_pos_approx, d_pos_approx)
                            k += 1
                        if d < nearest_drop_dist:
                            nearest_drop_dist = d
                            nearest_drop_node = d_pos_approx
                    if self.VERBOSE:
                        print(f"      path to d_id: {did} - dist: {d}")

                total_trip_cost = dist_to_coin + nearest_drop_dist

                if total_trip_cost < best_cost:
                    best_cost = total_trip_cost
                    best_goal_id = g_id
                    best_goal_node = g_pos_approx
                    best_drop_node = nearest_drop_node

            if best_goal_id is None:
                if GlobalConfig.GLOBAL_PLANNER_VERBOSE:
                    print(f"[GlobalPlanner] Warning: Could not reach remaining goals with {curr_agent_name}")
                break

            # 3. Assign Task
            curr_agent["workload"] += best_cost
            curr_agent["start_node"] = best_drop_node

            # Extend plan (avoiding duplicates at connection points)
            curr_agent["plan"].append(best_goal_node)
            curr_agent["plan"].append(best_drop_node)

            unassigned_goals.pop(best_goal_id)

        agent_goals: Dict[str, List] = {agent_name: (data["plan"]) for agent_name, data in agent_states.items()}
        agent_workload: Dict[str, float] = {agent_name: data["workload"] for agent_name, data in agent_states.items()}

        return agent_goals, agent_workload
