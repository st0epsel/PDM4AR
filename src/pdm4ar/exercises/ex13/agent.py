from dataclasses import dataclass
from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from pdm4ar.exercises.ex13.planner import SatellitePlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj
from pdm4ar.exercises.ex13.config import Config

import numpy as np


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    # Controller gains: P, I, D, AntiWindup saturation
    pid_pos: tuple = (2.7, 0.0, 2.0, 100.0)
    pid_heading: tuple = (5.25, 5.0, 0.0, 100.0)
    pid_vel: tuple = (8.0, 10.2, 0, 4.0)
    pid_w: tuple = (11.8, 7.6, 0, 15.0)
    # #passed number 1 and 2 wih kx 0.8 kv 1.0 kpsi 1.5 kdpsi 0.3 -> score 902, 872; margin is 0.2


class SatelliteAgent(Agent):
    # How does it enter in the simulation? The SpaceshipAgent object is created as value
    # corresponding to key "PDM4ARSpaceship" in dict "players", which is an attribute of
    # SimContext returned by "sim_context_from_yaml" in utils_config.py #forpush
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """
    last_t: float
    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.agent_params = MyAgentParams()

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # ... (Your existing setup code) ...
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.goal_state = init_sim_obs.goal.target
        self.planner = SatellitePlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
        )
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)

        # Setting start and end coordinates
        X_start = self.init_state.as_ndarray()
        X_end = self.goal_state.as_ndarray()

        # Input must be 0 at start and end according to task definition.
        U_start = np.array([[0.0], [0.0]])
        U_end = np.array([[0.0], [0.0]])

        # Computation of a physics based guess for p
        distance = np.sqrt((X_start[0] - X_end[0]) ** 2 + (X_start[1] - X_end[1]) ** 2)

        # CHANGED: Force these parameters to be standard floats immediately
        mass = float(self.sp.m_v)
        min_val, max_val = self.sp.F_limits
        force = float(2 * max(abs(min_val), abs(max_val)))

        # CHANGED: Ensure time is a float
        straight_line_time = float(np.sqrt(2 * distance * mass / (force)))

        # CHANGED: Ensure p is a float.
        # If this remains a SymPy type, it crashes the DiscretizationMethod.
        p_initial_guess = float(2 * straight_line_time)

        K = self.planner.params.K

        # CHANGED: Explicitly cast numpy arrays to float type
        # (This removes any lingering object/symbolic types from the linspace)
        self.planner.X_bar = np.linspace(np.squeeze(X_start), np.squeeze(X_end), num=K, axis=1).astype(float)
        self.planner.U_bar = np.linspace(np.squeeze(U_start), np.squeeze(U_end), num=K, axis=1).astype(float)
        self.planner.p_bar = np.array([p_initial_guess], dtype=float)  # Wrap in array just to be safe

        self.planner.X_bar[3, :] = 0.0
        self.planner.X_bar[4, :] = 0.0

        # handle angular orientation
        delta_psi = self.goal_state.psi - self.init_state.psi
        if abs(delta_psi) > np.pi:
            psi_start = self.init_state.psi
            psi_goal = self.goal_state.psi

            if delta_psi > np.pi:
                psi_goal_equivalent = psi_goal - 2 * np.pi
            else:
                psi_goal_equivalent = psi_goal + 2 * np.pi

            psi_index = 2

            # CHANGED: Ensure the angular row is also cast to float
            psi_row = np.linspace(psi_start, psi_goal_equivalent, num=K).astype(float)
            self.planner.X_bar[psi_index, :] = psi_row % (2 * np.pi)

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

        self.controller = SatelliteController(
            mass=self.sp.m_v,
            inertia=self.sg.Iz,
            width=self.sg.width,
            pid_pos=self.agent_params.pid_pos,
            pid_heading=self.agent_params.pid_heading,
            pid_vel=self.agent_params.pid_vel,
            pid_w=self.agent_params.pid_w,
        )

        self.last_t = 0.0

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:

        t_now = float(sim_obs.time)
        dt = t_now - self.last_t
        x_act = sim_obs.players[self.myname].state
        self.actual_trajectory.append(x_act)

        # Optional: Visualise trajectory every 1.0 seconds
        if int(sim_obs.time * 10) % 40 == 0:
            if Config.AGENT_PLOT:
                filename = f"{Config.EPISODE_NAME}_{(np.abs(self.init_state.y)%1*100):.0f}_{(10*sim_obs.time):.0f}.png"
                plot_traj(
                    computed=self.state_traj,
                    filename=filename,
                    actual=self.actual_trajectory,
                    sub_dir=Config.AGENT_PLOT_SUBFOLDER,
                )
            if Config.AGENT_VERBOSE:
                print(f"current sim_obs.time: {sim_obs.time}")
                print("current state is ", x_act)

        cmds = self.cmds_plan.at_interp(sim_obs.time)

        x_des = self.state_traj.at_interp(sim_obs.time)

        if Config.AGENT_VERBOSE:
            ex = x_act.x - x_des.x
            ey = x_act.y - x_des.y
            error = np.sqrt(ex**2 + ey**2)
            proj = error * np.cos(x_act.psi - np.pi + np.arctan2(ey, ex))
            print(
                f"is at {x_act.x, x_act.y}, should be at {x_des.x, x_des.y}, error: {x_act.x-x_des.x, x_act.y-x_des.y, proj}"
            )

        dist = np.sqrt((x_act.x - self.goal_state.x) ** 2 + (x_act.y - self.goal_state.y) ** 2)
        if Config.AGENT_VERBOSE and dist < 0.2:
            print(f"state is {x_act}, should be {self.goal_state}")

        if Config.PID:
            F_L_ref = cmds.F_left
            F_R_ref = cmds.F_right

            PID_cmds = self.controller.compute_control(current=x_act, ref_state=x_des, ref_control=cmds, dt=dt)

            F_L = PID_cmds[0]
            F_R = PID_cmds[1]
            self.last_t = t_now

        else:
            F_L = cmds.F_left
            F_R = cmds.F_right

        return SatelliteCommands(F_left=F_L, F_right=F_R)


class SatelliteController:

    class PID:
        # Helper class to store PID gains, integrators and derivatives
        def __init__(self, k):
            if len(k) != 4:
                raise ValueError("PID gains must be a tuple of 4 elements: (kp, ki, kd, sat)")
            self.kp = k[0]
            self.ki = k[1]
            self.kd = k[2]
            self.sat = k[3]
            self.prev_error = 0
            self.integral = 0

        def update(self, error, dt):
            self.integral += error * dt
            max_integral = self.sat / self.ki if self.ki != 0 else float("inf")
            if Config.PID_VERBOSE:
                if self.integral > max_integral or self.integral < -max_integral:
                    print(f"integral clipped (before: {self.integral})")
            self.integral_clipped = np.clip(self.integral, -max_integral, max_integral)
            p_term = self.kp * error
            i_term = self.ki * self.integral_clipped
            derivative = (error - self.prev_error) / dt if dt > 0 else 0
            d_term = self.kd * derivative

            output = p_term + i_term + d_term

            self.prev_error = error
            if Config.PID_VERBOSE:
                if output > self.sat or output < -self.sat:
                    print(f"output clipped (before : {output})")
            output = np.clip(output, -self.sat, self.sat)

            return output

        def reset(self):
            self.integral = 0
            self.prev_error = 0

    def __init__(self, mass, inertia, width, pid_pos, pid_heading, pid_vel, pid_w):
        self.m = mass
        self.Iz = inertia
        self.l = width

        # Surge Controller (Controls Fx)
        self.pid_surge_pos = self.PID(pid_pos)
        self.pid_surge_vel = self.PID(pid_vel)
        # Steering/Yaw Controller (Controls Torque)
        self.pid_heading = self.PID(pid_heading)
        self.pid_w = self.PID(pid_w)

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.pid_surge_pos.reset()
        self.pid_surge_vel.reset()
        self.pid_heading.reset()
        self.pid_w.reset()

    def compute_control(
        self, current: SatelliteState, ref_state: SatelliteState, ref_control: SatelliteCommands, dt: float
    ) -> tuple[float, float, float, float]:
        """
        Compute the control inputs for the satellite using a cascaded PID controller with feedforward.

        Args:
            current (SatelliteState):       ... The current state of the satellite.
            ref_state (SatelliteState):     ... The reference state of the satellite.
            ref_control SatelliteCommands:              ... The reference angular acceleration.
            dt (float):                     ... Time step for PID updates.

        Returns:
            F_L (float):                    ... Force command for the left thruster.
            F_R (float):                    ... Force command for the right thruster.
            acc_surge_ref (float):          ... Implied surge acceleration from feedforward.
            alpha_yaw_ref (float):          ... Implied yaw angular acceleration from feedforward.
        """
        # --- A. DECODE FEEDFORWARD FORCES ---
        # We assume the reference inputs are "Perfect" for the nominal plant

        # 1. Calculate Feedforward Body Force & Moment
        F_ff = ref_control.F_left + ref_control.F_right
        M_ff = (ref_control.F_right - ref_control.F_left) * (self.l / 2.0)

        # 2. Determine implied Acceleration
        # Useful for telemetry to know what the reference "thinks" is happening
        acc_surge_ref = F_ff / self.m
        alpha_yaw_ref = M_ff / self.Iz

        # --- B. FEEDBACK LOOPS ---
        # 1. Error Calc (Global -> Body)
        dx = ref_state.x - current.x
        dy = ref_state.y - current.y
        cs, sn = np.cos(current.psi), np.sin(current.psi)
        ex_b = dx * cs + dy * sn
        ey_b = -dx * sn + dy * cs

        # 2. Surge Control
        vx_correction = self.pid_surge_pos.update(ex_b, dt)
        vx_target = ref_state.vx + vx_correction
        ev_x = vx_target - current.vx
        F_fb = self.pid_surge_vel.update(ev_x, dt)  # Feedback Force

        # 3. Yaw Control (Slip Aware)
        steering_gain = 0.5
        heading_correction = np.arctan(steering_gain * ey_b)
        theta_target = ref_state.psi + heading_correction
        e_theta = self.wrap_angle(theta_target - current.psi)
        w_correction = self.pid_heading.update(e_theta, dt)
        w_target = ref_state.dpsi + w_correction
        e_w = w_target - current.dpsi
        M_fb = self.pid_w.update(e_w, dt)  # Feedback Moment

        # --- C. COMBINATION & ALLOCATION ---
        F_total = F_ff + F_fb
        M_total = M_ff + M_fb

        F_L_cmd = (F_total / 2.0) - (M_total / self.l)
        F_R_cmd = (F_total / 2.0) + (M_total / self.l)

        # Return commands + the derived accelerations for logging
        return (F_L_cmd, F_R_cmd, acc_surge_ref, alpha_yaw_ref)
