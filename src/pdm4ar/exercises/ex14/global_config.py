from dataclasses import dataclass
from pickle import TRUE
import os
from pdm4ar.exercises_def.structures import out_dir


@dataclass(frozen=True)
class GlobalConfig:

    # TUNIGN - GLOBAL PLANNER
    LOOK_CLOSE_TO_UNREACHABLE_NODES = False  # If a dropoff / coin node is within the inflated boundaries, try to reach a node close to it to still get points at the risk of loosing slightly

    # TUNING - GRAPH
    CRASH_BUFFER = 0.15  # [m] Additional buffer around obstacles to avoid crashes
    N_NODES = 900  # Number of graph nodes
    NODE_CONNECTION_MULT = 1.4  # [m] Max Node connection radius in relation to average node distance (calculated from n_nodes and map area)
    RELAX_GRAPH = True  # Enable graph relaxation to increase connectivity

    # TUNING - AGENTS
    RETREAT_DIST = 1.4  # [m]
    NOT_YIELDING_AGNET_BUFFER = 2.7  # * robot.radius to determine buffer zone where Graph nodes are deleted if self has lower priority than other
    MIN_COL_RADIUS: float = (
        0.4  # [m]  # Minimum collision radius between two agents before drastic measures (replanning/retreat) are taken
    )
    ABORT_RETREAT: float = 0.7  # [m]  # Retreat hysteresis minimum distance
    SMOOTH_PATH = True  # Make path a cubic spline
    REDUCE_PATH_NODES = True  # Linearize path between obstacles to be straight lines

    # TUNING - CONTROLLER
    FF_GAIN = 0.6  # Feedforward gain
    FF_DIST = 0.4  # [m] Feedforward lookahead distance
    STATE_ERROR_GAIN = 4.1  # State error proportional gain

    # OUTPUT CONTROL
    SESSION_NAME = "cosmetic_test"

    ALL_PLAYERS_PERF_METRIC = True
    PLAYER_PERF_METRIC = True

    VID = False

    PLOT_GRAPH_SUBDIR = "Graph"
    PLOT_RELAXATION = False
    PLOT_AGENT_PLANS = False

    GLOBAL_PLANNER_VERBOSE = False
    AGENT_VERBOSE = False
    GRAPH_VERBOSE = False
    OUTPUT_VERBOSE = False

    out_folder = os.path.join(out_dir("14"), "index.html_resources")
    id = 0
    while os.path.isdir(os.path.join(out_folder, f"{SESSION_NAME}_{id}")):
        id += 1

    SESSION_FOLDER = os.path.join(out_folder, f"{SESSION_NAME}_{id}")
