from dataclasses import dataclass
import os
from tkinter import SE

from pdm4ar.exercises_def.structures import out_dir


@dataclass(frozen=True)
class Config:

    # Get Performance Details
    DYNAMICS_CHECK = False
    PERF_METRIC = False  # Outputs details on lost points

    # Plot & Video Generation
    EPISODE_NAME = "nIGs"  # File Name Prefix

    AGENT_PLOT = True  # Activates plot generation
    AGENT_PLOT_SUBFOLDER = "AgentPlot"

    PLANNER_PLOT = True  # Activates plot generation during trajectory optimization
    PLANNER_PLOT_SUBFOLDER = "PlannerPlot"

    VID = False  # Activates Video generation

    # Controlls printoutputs
    PLOT_VERBOSE = False
    SOLVER_VERBOSE = False  # Sets Solver to Verbose
    AGENT_VERBOSE = False  # Sets Agent to Verbose
    PLANNER_VERBOSE = False  # Sets Planner to Verbose
    PID_VERBOSE = False

    UNNECESSARY_VARIABLE = False  # only here for sumbit purpose, delete later

    PID = False  # Disables PID Controll

    SESSION_NAME = "MAYBE_THIS_THYME"

    out_folder = os.path.join(out_dir("14"), "index.html_resources")
    id = 0
    while os.path.isdir(os.path.join(out_folder, f"{SESSION_NAME}_{id}")):
        id += 1

    SESSION_FOLDER = os.path.join(out_folder, f"{SESSION_NAME}_{id}")
