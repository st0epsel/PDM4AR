import ast
from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj

from pdm4ar.exercises_def.structures import out_dir

from pdm4ar.exercises.ex13.config import Config

import numpy as np


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 30  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e6  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 10  # initial trust region radius
    min_tr_radius: float = 0.01  # min trust region radius
    max_tr_radius: float = 50  # max trust region radius
    rho_0: float = 0.01  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 10  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant

    safety_margin = 0.2
    # 0.3 still passes the local tests and the simi_tests. However it gets quite close to a planet in test case 2.
    # 0.1 results in several collisions
    # used to be 1.0 or 1.2 depending on where you look in the implementation


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(  # This is the constructor, variables only get initialised here.
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        self.num_obstacles = len(self.planets) + len(self.asteroids)

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        if Config.DYNAMICS_CHECK:
            # Check dynamics implementation (pass this test before going further. It is not part of the final evaluation, so you can comment it out later)
            if not self.integrator.check_dynamics():
                raise ValueError("Dynamics check failed.")
            else:
                print("Dynamics check passed.")

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    """def generate_initial_guesses(self, init_state: SatelliteState, goal_state: DynObstacleState):
        X_bar, U_bar, p_bar = self.initial_guess()
        return ((X_bar, U_bar, p_bar),)"""

    def generate_initial_guesses(self, init_state: SatelliteState, goal_state: DynObstacleState):
        """
        Generates N+1 initial path guesses for N planets using a 'sandwich' topology approach.
        Uses numpy polyfit (quadratic) instead of scipy splines to avoid external dependencies.
        Orientation is linearly interpolated; velocities are set to zero.
        """
        # 1. Setup & Data Extraction
        IGs = []
        K = self.params.K
        n_x, n_u = self.satellite.n_x, self.satellite.n_u

        # Extract positions
        start_pos = np.array([init_state.x, init_state.y])
        g_arr = goal_state.as_ndarray()
        end_pos = np.array([g_arr[0], g_arr[1]])

        # 2. Coordinate System Transform (Align Start->Goal with +X axis)
        diff = end_pos - start_pos
        dist_total = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0])

        # Rotation Matrices
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])  # Global -> Local
        R_inv = np.array([[c, s], [-s, c]])  # Local -> Global

        # 3. Process Planets (Transform to local frame)
        local_planets = []
        # Use self.planets dictionary provided in __init__
        for p in self.planets.values():
            # Planet center in global frame
            p_center = np.array(p.center)
            # Transform to local frame relative to start_pos
            p_loc = R @ (p_center - start_pos)

            local_planets.append(
                {
                    "y": p_loc[1],  # Vertical offset (key for sorting)
                    "x": p_loc[0],  # Horizontal distance
                    "r": float(p.radius),
                }
            )

        # Sort planets by their vertical position (Y) to identify distinct gaps
        local_planets.sort(key=lambda item: item["y"])

        # 4. Generate N+1 Paths
        num_paths = len(local_planets) + 1
        margin = 2.0  # Heuristic buffer

        for i in range(num_paths):
            # Define 3 Key Points: Start (0,0), Midpoint (Gap), End (dist, 0)
            pts_x = [0.0]
            pts_y = [0.0]

            # --- Determine the Gap Midpoint ---
            if i == 0:  # BELOW lowest planet
                if local_planets:
                    p = local_planets[0]
                    pts_x.append(p["x"])
                    pts_y.append(p["y"] - (p["r"] + margin))

            elif i == len(local_planets):  # ABOVE highest planet
                if local_planets:
                    p = local_planets[-1]
                    pts_x.append(p["x"])
                    pts_y.append(p["y"] + (p["r"] + margin))

            else:  # BETWEEN Planet i-1 and Planet i
                p_bot = local_planets[i - 1]
                p_top = local_planets[i]

                mid_x = (p_bot["x"] + p_top["x"]) / 2

                # Midpoint of the vertical gap
                y_bot_edge = p_bot["y"] + p_bot["r"]
                y_top_edge = p_top["y"] - p_top["r"]
                mid_y = (y_bot_edge + y_top_edge) / 2

                pts_x.append(mid_x)
                pts_y.append(mid_y)

            # Add Goal Point
            pts_x.append(dist_total)
            pts_y.append(0.0)

            # --- Polynomial Fitting (Quadratic) ---
            # Fit a parabola (deg=2) through the 3 points
            # This replaces CubicSpline while keeping smooth curvature
            coeffs = np.polyfit(pts_x, pts_y, 2)

            # Generate local path
            xs_local = np.linspace(0, dist_total, K)
            ys_local = np.polyval(coeffs, xs_local)

            # --- Construct State Matrix X ---
            X_guess = np.zeros((n_x, K))

            # Linear Interpolation for Psi (Orientation)
            # Assuming g_arr structure matches SatelliteState [x, y, psi, vx, vy, dpsi]
            start_psi = init_state.psi
            end_psi = g_arr[2]
            psi_interp = np.linspace(start_psi, end_psi, K)

            for k in range(K):
                # 1. Transform Position back to Global
                lx, ly = xs_local[k], ys_local[k]
                pos_global = (R_inv @ np.array([lx, ly])) + start_pos

                # 2. Fill State
                X_guess[0, k] = pos_global[0]  # x
                X_guess[1, k] = pos_global[1]  # y
                X_guess[2, k] = psi_interp[k]  # psi (Linear Interpolation)
                X_guess[3, k] = 0.0  # vx (Set to 0)
                X_guess[4, k] = 0.0  # vy (Set to 0)
                X_guess[5, k] = 0.0  # dpsi (Set to 0)

            # Enforce exact Start/Goal Position and Orientation
            X_guess[0:3, 0] = [init_state.x, init_state.y, init_state.psi]
            X_guess[0:3, -1] = [g_arr[0], g_arr[1], g_arr[2]]

            # --- Inputs U ---
            U_guess = np.zeros((n_u, K))

            # Time Estimate
            avg_vel = 1.0
            p_guess = np.array([max(10.0, dist_total / avg_vel)])

            IGs.append((X_guess, U_guess, p_guess))

        # --- Visualize ---
        self._visualize_guesses(IGs, init_state, goal_state)

        return tuple(IGs)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:

        if Config.PLANNER_VERBOSE:
            print(f"\n=== STARTING SCVX PLANNER ===")
        self.init_state = init_state
        self.goal_state = goal_state

        IGs = self.generate_initial_guesses(init_state=init_state, goal_state=goal_state)
        IG_final_scores = []
        IG_results = []

        for n_IG, IG in enumerate(IGs):
            if Config.PLANNER_VERBOSE:
                print(f" --- Starting initial guess {n_IG+1} of {len(IGs)} --- ")
            # 1. Setup Parameters
            self.problem_parameters["init_state"].value = init_state.as_ndarray()
            self.problem_parameters["goal_state"].value = goal_state.as_ndarray()
            self.problem_parameters["init_control"].value = np.zeros(self.satellite.n_u)
            self.problem_parameters["goal_control"].value = np.zeros(self.satellite.n_u)
            self.problem_parameters["tr_radius"].value = self.params.tr_radius

            # Initial Trajectory
            self.X_bar, self.U_bar, self.p_bar = IG

            # Setup weights
            base_lambda = self.params.lambda_nu
            weights = np.array([base_lambda] * self.satellite.n_x)
            self.problem_parameters["weight_nu"].value = weights

            X_integrated = self.integrator.integrate_nonlinear_piecewise(self.X_bar, self.U_bar, self.p_bar)

            # 1. Time Cost
            c_time = (self.params.weight_p @ self.p_bar).item()
            # 2. Dynamics Defect Cost
            c_defect = self.params.lambda_nu * np.sum(np.abs(self.X_bar - X_integrated))
            # 3. Goal Cost
            c_goal = 1e5 * np.sum(np.abs(self.X_bar[:, -1] - goal_state.as_ndarray()))

            # 4. Obstacle Cost (FIXED: Calculate true violation)
            c_obs = self._get_true_obstacle_cost(self.X_bar, self.p_bar[0])

            self.last_nonlinear_cost = c_time + c_defect + c_goal + c_obs

            if Config.PLANNER_VERBOSE:
                print(
                    f"  [DEBUG] Init Cost: {self.last_nonlinear_cost:.2e} (Time: {c_time:.1f}, Defect: {c_defect:.1e}, Goal: {c_goal:.1e}, Obs: {c_obs:.1e})"
                )

            # 3. Main Loop
            for it in range(self.params.max_iterations):

                if Config.PLANNER_PLOT:
                    # Plot current path
                    state_traj = self._extract_seq_from_array(self.X_bar, self.U_bar, self.p_bar)[1]
                    filename = f"{Config.EPISODE_NAME}_{(np.abs(self.init_state.y)%1*100):.0f}_{n_IG}_it{it}.png"
                    plot_traj(computed=state_traj, filename=filename, sub_dir=Config.PLANNER_PLOT_SUBFOLDER)

                self._convexification()

                try:
                    self.problem.solve(solver=self.params.solver, verbose=False)
                except cvx.SolverError as e:
                    print(f"  [ERROR] Iter {it}: Solver crashed: {e}")
                    self._update_trust_region(0.0)
                    continue

                if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"  [ERROR] Iter {it}: Solver status '{self.problem.status}'")
                    if self.problem.status == "infeasible":
                        break
                    self._update_trust_region(0.0)
                    continue

                # Extract new values
                X_new = self.variables["X"].value
                U_new = self.variables["U"].value
                p_new = self.variables["p"].value
                nu_new = self.variables["nu"].value
                s_obs_new = self.variables["s_obs"].value

                if X_new is None or p_new is None:
                    self._update_trust_region(0.0)
                    continue

                # --- CALCULATE COSTS (Must match _get_objective EXACTLY) ---

                # Common Terms
                cost_time = (self.params.weight_p @ p_new).item()
                cost_goal = 1e5 * np.sum(np.abs(X_new[:, -1] - goal_state.as_ndarray()))

                # --- LINEAR COSTS (For Predicted Reduction) ---
                # Use solver slack for linear cost
                cost_nu_linear = self.params.lambda_nu * np.sum(np.abs(nu_new))
                cost_obs_linear = 1e5 * np.sum(s_obs_new)

                linear_total = cost_time + cost_nu_linear + cost_goal + cost_obs_linear

                # --- NONLINEAR COSTS (For Actual Reduction) ---
                # 1. Integrate dynamics
                X_int = self.integrator.integrate_nonlinear_piecewise(X_new, U_new, p_new)
                cost_nu_nonlinear = self.params.lambda_nu * np.sum(np.abs(X_new - X_int))
                cost_obs_nonlinear = self._get_true_obstacle_cost(X_new, p_new[0])

                # 2. Calculate TRUE obstacle violation (FIXED)
                # This calls the helper method to get the real distance to obstacles

                nonlinear_total = cost_time + cost_nu_nonlinear + cost_goal + cost_obs_nonlinear

                # Rho Calculation
                actual_red = self.last_nonlinear_cost - nonlinear_total
                pred_red = self.last_nonlinear_cost - linear_total

                # Avoid division by zero
                if abs(pred_red) < 1e-6:
                    rho = 0.0
                else:
                    rho = actual_red / pred_red

                if Config.PLANNER_VERBOSE:
                    print(f"  Iter {it}: Rho={rho:.2f} | p={p_new[0]:.2f}s | Slack={np.max(np.abs(nu_new)):.2e}")

                if rho > self.params.rho_0:
                    X_nl = self.integrator.integrate_nonlinear_piecewise(X_new, U_new, p_new)
                    self.X_bar = X_nl
                    self.U_bar = U_new
                    self.p_bar = p_new
                    cost_nu_nonlinear = self.params.lambda_nu * np.sum(
                        np.abs(X_nl - self.integrator.integrate_nonlinear_piecewise(X_nl, U_new, p_new))
                    )

                    self.last_nonlinear_cost = nonlinear_total
                    self._update_trust_region(rho)
                else:
                    self._update_trust_region(rho)

                if self.problem_parameters["tr_radius"].value < self.params.min_tr_radius:
                    print(f"=== NOT Converged - bad initial guess ===")
                    break

                if self._check_convergence():
                    if Config.PLANNER_VERBOSE:
                        print(f"=== Converged after {it} iterations ===")
                    break

            # mycmds, mystates = self._extract_seq_from_array(self.X_bar, self.U_bar, self.p_bar)
            output = self._extract_seq_from_array(self.X_bar, self.U_bar, self.p_bar)
            IG_results.append(output)
            IG_final_scores.append(linear_total)

        if Config.PLANNER_VERBOSE:
            print(f"=== PLANNER FINISHED ===")

        IG_final_scores = np.array(IG_final_scores)
        best_IG = np.argmin(IG_final_scores)
        if Config.PLANNER_VERBOSE:
            print(f"    Best initial guess: NR{best_IG}")

        mycmds, mystates = IG_results[best_IG]

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        K = self.params.K

        # Linear interpolation from Start to Goal
        # NOTE: Using the internal stored state/goal if available
        if hasattr(self, "init_state") and hasattr(self, "goal_state"):
            x0 = self.init_state.as_ndarray()
            xf = self.goal_state.as_ndarray()
        else:
            # Fallback (should ideally not happen if called correctly)
            x0 = np.zeros(self.satellite.n_x)
            xf = np.zeros(self.satellite.n_x)

        X = np.linspace(x0, xf, K).T
        U = np.zeros((self.satellite.n_u, K))

        # Use a more conservative time estimate (slower is safer for avoidance)
        dist = np.linalg.norm(x0[0:2] - xf[0:2])
        avg_vel = 0.5  # Assume conservative speed
        p = np.array([max(10.0, dist / avg_vel)])

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """

        variables = {
            "X": cvx.Variable((self.satellite.n_x, self.params.K)),
            "U": cvx.Variable((self.satellite.n_u, self.params.K)),
            "p": cvx.Variable(self.satellite.n_p),
            "nu": cvx.Variable((self.satellite.n_x, self.params.K - 1)),  # slack variables
            "s_obs": cvx.Variable((max(1, self.num_obstacles), self.params.K - 1), nonneg=True),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K

        # 1. Dynamics Parameters (Lists for each time step)
        A = [cvx.Parameter((n_x, n_x)) for _ in range(K - 1)]

        B_plus = [cvx.Parameter((n_x, n_u)) for _ in range(K - 1)]
        B_minus = [cvx.Parameter((n_x, n_u)) for _ in range(K - 1)]

        F = [cvx.Parameter((n_x, n_p)) for _ in range(K - 1)]
        r = [cvx.Parameter((n_x,)) for _ in range(K - 1)]

        # 2. Obstacle Parameters
        # We need to store the Position (P) and Normal Vector (H) for every obstacle at every time step
        num_obstacles = len(self.planets) + len(self.asteroids)

        # Structure: List[List[Parameter]] -> [Obstacle_i][Time_k]
        obs_H = [[cvx.Parameter(2) for _ in range(K - 1)] for _ in range(num_obstacles)]
        obs_P = [[cvx.Parameter(2) for _ in range(K - 1)] for _ in range(num_obstacles)]

        # Stores the scalar value: n^T * v_ast * (k / (K-1))

        obs_grad_term = [[cvx.Parameter() for _ in range(K - 1)] for _ in range(num_obstacles)]

        # Scalar parameter to hold (H * P - grad * prev_p)
        obs_const = [[cvx.Parameter() for _ in range(K - 1)] for _ in range(num_obstacles)]

        problem_parameters = {
            "init_state": cvx.Parameter(n_x),
            "init_control": cvx.Parameter(n_u),
            "goal_state": cvx.Parameter(n_x),
            "goal_control": cvx.Parameter(n_u),
            "prev_X": cvx.Parameter((n_x, K)),
            "prev_U": cvx.Parameter((n_u, K)),
            "prev_p": cvx.Parameter(n_p),
            "A": A,
            "B_plus": B_plus,
            "B_minus": B_minus,
            "F": F,
            "r": r,
            # Obstacle Parameters
            "obs_H": obs_H,
            "obs_P": obs_P,
            "obs_grad_term": obs_grad_term,
            "obs_const": obs_const,
            "tr_radius": cvx.Parameter(nonneg=True),
            "weight_nu": cvx.Parameter(n_x, nonneg=True),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]
        par = self.problem_parameters
        K = self.params.K

        x_min, x_max = -11.0, 11.0
        y_min, y_max = -11.0, 11.0
        margin = 1.5

        constraints = [
            X[:, 0] == par["init_state"],
            X[:, -1] == par["goal_state"],
            U[:, 0] == par["init_control"],
            U[:, -1] == par["goal_control"],
            p >= 0.1,
        ]

        X = self.variables["X"]  # shape (6, K)
        x_pos = X[0, :]  # all x positions
        y_pos = X[1, :]  # all y positions

        constraints += [
            x_pos >= x_min + margin,
            x_pos <= x_max - margin,
            y_pos >= y_min + margin,
            y_pos <= y_max - margin,
        ]

        # Dynamics Loop
        for k in range(K - 1):
            constraints.append(
                X[:, k + 1]
                == par["A"][k] @ X[:, k]
                + par["B_minus"][k] @ U[:, k]  # Effect of current input
                + par["B_plus"][k] @ U[:, k + 1]  # Effect of next input
                + par["F"][k] @ p
                + par["r"][k]
                + nu[:, k]
            )

        # ---------------------------------------------------------
        # TRUST REGION (Paper Implementation - Eq 51g & 46)
        # ---------------------------------------------------------
        # We define deviations from the linearization point
        dx = X - par["prev_X"]
        du = U - par["prev_U"]
        dp = p - par["prev_p"]

        # SCALING FACTORS (To prevent Position from overwhelming Thrust)
        # Adjust these based on your problem magnitude.
        # Example: If X is ~10.0 and U is ~1.0, weigh X less or U more.
        w_pos = 1.0  # Weight for Position/Velocity (assuming ~10m range)
        w_u = 1.0  # Weight for Thrust (assuming ~1-2N range)
        w_p = 1.0  # Weight for Time

        # We use L1 Norm (Sum of absolute values) as it is robust and linear (q=1)
        # You can also use L2 (cvx.norm(..., 2)) but L1 is often faster for these problems.

        tr_radius = par["tr_radius"]

        # Enforce Eq 51g at EVERY time step k
        for k in range(K):
            # Calculate the weighted norm sum for this specific time step
            # ||dx_k|| + ||du_k|| <= eta

            # Note: dx[:, k] is the state vector at step k
            state_norm = w_pos * cvx.norm(dx[:, k], 1)
            input_norm = w_u * cvx.norm(du[:, k], 1)

            # The Trust Region constraint
            constraints.append(state_norm + input_norm + w_p * cvx.norm(dp, 1) <= tr_radius)

        # Input Limits
        f_min, f_max = self.sp.F_limits
        constraints.append(U >= 0.95 * f_min)
        constraints.append(U <= 0.95 * f_max)

        # --- OBSTACLE AVOIDANCE ---
        # Gather all obstacle objects just to get their Radius and Count

        # Use consistent safety margin (make sure it matches _get_true_obstacle_cost)

        # DELETE THE FIRST OBSTACLE LOOP HERE

        # KEEP ONLY THIS LOOP (The one with dP_dp)

        all_obstacles = []
        if self.planets:
            all_obstacles.extend(self.planets.values())
        if self.asteroids:
            all_obstacles.extend(self.asteroids.values())

        # Use consistent safety margin (make sure it matches _get_true_obstacle_cost)
        r_sat = 1.4
        s_obs = self.variables["s_obs"]

        # Obstacle loop
        for k in range(K - 1):
            p_k = X[0:2, k + 1]

            for i, obs in enumerate(all_obstacles):
                r_obs = float(obs.radius)
                r_total = r_sat + r_obs + self.params.safety_margin

                H_vec = par["obs_H"][i][k]

                # Get Gradient
                grad_term = par["obs_grad_term"][i][k]

                obs_c = par["obs_const"][i][k]

                # Dynamic Constraint
                constraints.append(H_vec @ p_k - grad_term * p - obs_c >= r_total - s_obs[i, k])

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # 1. Minimize Final Time
        J_original = self.params.weight_p @ self.variables["p"]

        # 2. Minimize Slack (Dynamics Violation)
        # We use matrix multiplication (weight_nu.T @ |nu|) to sum the weighted errors
        # weight_nu is (n_x,), nu is (n_x, K-1).
        # The result of weight_nu @ abs(nu) is a vector of size (K-1,), we sum that.
        nu_abs = cvx.abs(self.variables["nu"])
        J_penalty = cvx.sum(self.problem_parameters["weight_nu"] @ nu_abs)
        J_input = 2 * cvx.sum(cvx.abs(self.variables["U"]))
        s_obs = self.variables["s_obs"]
        J_obs = 1e6 * cvx.sum(s_obs)

        return cvx.Minimize(J_original + J_penalty + J_input + J_obs)

    def _convexification(self):
        """
        Perform convexification step: Linearization, Discretization.
        """
        # 1. Update Previous Guess
        self.problem_parameters["prev_X"].value = self.X_bar
        self.problem_parameters["prev_U"].value = self.U_bar
        self.problem_parameters["prev_p"].value = self.p_bar

        n_x, n_u, n_p = self.satellite.n_x, self.satellite.n_u, self.satellite.n_p
        K = self.params.K

        # 2. Get Linearization Matrices (A, B, F)
        # We integrate the CURRENT guess to get the reference trajectory
        X_nl = self.integrator.integrate_nonlinear_piecewise(self.X_bar, self.U_bar, self.p_bar)

        disc_output = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        # Handle different return types (FOH vs ZOH)
        if len(disc_output) == 5:
            A_flat, B_plus_flat, B_minus_flat, F_flat, _ = disc_output  # Ignore 'r'
        elif len(disc_output) == 4:
            A_flat, B_flat, F_flat, _ = disc_output  # Ignore 'r'
            B_minus_flat = B_flat
            B_plus_flat = np.zeros_like(B_flat)

        Ak = A_flat.reshape((n_x, n_x, K - 1), order="F")
        Bk_plus = B_plus_flat.reshape((n_x, n_u, K - 1), order="F")
        Bk_minus = B_minus_flat.reshape((n_x, n_u, K - 1), order="F")
        Fk = F_flat.reshape((n_x, n_p, K - 1), order="F")

        for k in range(K - 1):
            self.problem_parameters["A"][k].value = Ak[:, :, k]
            self.problem_parameters["B_plus"][k].value = Bk_plus[:, :, k]
            self.problem_parameters["B_minus"][k].value = Bk_minus[:, :, k]
            self.problem_parameters["F"][k].value = Fk[:, :, k]

            # --- ROBUST 'r' CALCULATION ---
            # r = x_next_nonlinear - (Linear_Terms)
            # This forces the constraint to be satisfied EXACTLY at deviation=0

            x_next_nl = X_nl[:, k + 1]
            x_curr = self.X_bar[:, k]
            u_curr = self.U_bar[:, k]
            u_next = self.U_bar[:, k + 1]

            # Note: We do NOT include F*p here because we use F*(p-prev_p) in constraints
            linear_part = Ak[:, :, k] @ x_curr + Bk_minus[:, :, k] @ u_curr + Bk_plus[:, :, k] @ u_next

            # Calculate F * prev_p using NumPy
            term_F_prev_p = Fk[:, :, k] @ self.p_bar

            # Assign to r: r = x_nl - linear - F*prev_p
            self.problem_parameters["r"][k].value = x_next_nl - linear_part - term_F_prev_p

        # 3. Obstacle Geometry Calculation
        all_obstacles = []
        if self.planets:
            all_obstacles.extend(self.planets.values())
        if self.asteroids:
            all_obstacles.extend(self.asteroids.values())

        # Time step duration based on current guess
        dt = self.p_bar[0] / (K - 1)

        for k in range(K - 1):
            # Time at step k+1 (since we constrain X[k+1])
            t_k = (k + 1) * dt

            # Ship position guess
            p_ship_bar = self.X_bar[0:2, k + 1]

            for i, obs in enumerate(all_obstacles):
                # --- HANDLE ATTRIBUTES HERE ---

                # Case A: Planet (Static)
                if hasattr(obs, "center"):
                    p_obs = np.array(obs.center)

                # Case B: Asteroid (Dynamic)
                elif hasattr(obs, "start") and hasattr(obs, "velocity"):
                    start_pos = np.array(obs.start, dtype=float)

                    # velocity is given in the asteroid's LOCAL frame
                    vel_local = np.array(obs.velocity, dtype=float)
                    theta = float(getattr(obs, "orientation", 0.0))  # rad

                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -s], [s, c]])

                    # convert to GLOBAL frame
                    vel = R @ vel_local

                    # Predict position: P = P0 + V_global * t
                    p_obs = start_pos + vel * t_k

                # Case C: Fallback (simpler structure)
                elif hasattr(obs, "x"):
                    p_obs = np.array([float(obs.x), float(obs.y)])
                else:
                    p_obs = np.array([0.0, 0.0])  # Should not happen given your classes

                # --- CALCULATE NORMAL VECTOR ---
                dist_vec = p_ship_bar - p_obs
                dist_norm = np.linalg.norm(dist_vec)

                # IMPROVED: Handling 'Inside' Case
                if dist_norm < 1e-5:
                    # If we are exactly inside, the gradient is undefined.
                    # We pick a direction.
                    # Strategy 1: Perpendicular to velocity (try to step aside)
                    # Strategy 2: If dynamic, push away from velocity direction
                    if hasattr(obs, "velocity"):
                        # normalized velocity direction
                        v_norm = np.linalg.norm(vel)
                        if v_norm > 0:
                            # Push perpendicular to movement
                            n_vec = np.array([-vel[1], vel[0]]) / v_norm
                        else:
                            n_vec = np.array([1.0, 0.0])
                    else:
                        n_vec = np.array([1.0, 0.0])
                else:
                    n_vec = dist_vec / dist_norm

                # --- ASSIGN TO PARAMETERS ---
                self.problem_parameters["obs_H"][i][k].value = n_vec

        p_val = self.p_bar[0]

        for k in range(K - 1):
            # Normalized time tau (0 to 1)
            tau_k = (k + 1) / (K - 1)
            # Real time estimate
            t_k = tau_k * p_val

            for i, obs in enumerate(all_obstacles):
                # 1. Update Snapshot Position for dynamic obstacles
                if hasattr(obs, "velocity"):
                    start = np.array(obs.start, dtype=float)

                    # velocity in LOCAL frame -> rotate to GLOBAL frame
                    vel_local = np.array(obs.velocity, dtype=float)
                    theta = float(getattr(obs, "orientation", 0.0))
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -s], [s, c]])
                    vel = R @ vel_local

                    # position at time t_k
                    p_obs = start + vel * t_k

                    # 2. GRADIENT wrt final time p
                    # t_k = tau_k * p   => dP/dp = vel * tau_k
                    grad_p = vel * tau_k
                else:
                    # Static obstacles don't move when p changes
                    p_obs = (
                        np.array(obs.center, dtype=float)
                        if hasattr(obs, "center")
                        else np.array([obs.x, obs.y], dtype=float)
                    )
                    grad_p = np.zeros(2)

                self.problem_parameters["obs_P"][i][k].value = p_obs
                # This is the place where we make a change to ensure disciplined parameterised programming compliance
                n_vec = self.problem_parameters["obs_H"][i][k].value
                scalar_val = np.dot(n_vec, grad_p)
                self.problem_parameters["obs_grad_term"][i][k].value = scalar_val
                const_val = np.dot(n_vec, p_obs) - scalar_val * p_val
                self.problem_parameters["obs_const"][i][k].value = const_val

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        nu_val = self.variables["nu"].value
        X_val = self.variables["X"].value

        # 1. Check if dynamics are satisfied (slack is near zero)
        max_nu = np.max(np.abs(nu_val))

        # 2. Check if trajectory stopped changing
        delta_X = np.max(np.abs(X_val - self.X_bar))

        if Config.PLANNER_VERBOSE:
            print(f"  Convergence Check: Max Slack={max_nu:.2e}, Delta X={delta_X:.2e}")

        if max_nu < self.params.stop_crit and delta_X < self.params.stop_crit:
            return True

        return False

    def _update_trust_region(self, rho):
        """
        Update trust region radius.
        """
        curr_radius = self.problem_parameters["tr_radius"].value

        if rho < self.params.rho_0:
            # Very bad step. Shrink trust region.
            new_radius = curr_radius / self.params.alpha
            out = f"  Step Rejected (rho={rho:.2f}). Shrinking TR: {curr_radius:.2f} -> {new_radius:.2f}"
        elif rho < self.params.rho_1:
            # Marginal step. Shrink slightly.
            new_radius = curr_radius / self.params.alpha
            out = f"  Step Accepted (rho={rho:.2f}). Shrinking TR: {curr_radius:.2f} -> {new_radius:.2f}"
        elif rho < self.params.rho_2:
            # Good step. Keep radius.
            new_radius = curr_radius
            out = f"  Step Accepted (rho={rho:.2f}). Keeping TR: {curr_radius:.2f}"
        else:
            # Excellent step. Grow radius.
            new_radius = curr_radius * self.params.beta
            out = f"  Step Accepted (rho={rho:.2f}). Growing TR: {curr_radius:.2f} -> {new_radius:.2f}"

        if Config.PLANNER_VERBOSE:
            print(out)

        # Clip radius to limits
        new_radius = min(new_radius, self.params.max_tr_radius)

        self.problem_parameters["tr_radius"].value = new_radius

    # @staticmethod
    def _extract_seq_from_array(
        self,
        X_use: np.ndarray | None = None,
        U_use: np.ndarray | None = None,
        p_use: np.ndarray | None = None,
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        # Use provided arrays, otherwise fall back to CVX values #forpush
        if X_use is None:
            X_use = self.variables["X"].value
            U_use = self.variables["U"].value
            p_use = self.variables["p"].value

        if X_use is None or U_use is None or p_use is None:
            print("[ERROR] Optimization failed, returning trivial trajectory.")
            X_use = np.zeros((self.satellite.n_x, self.params.K))
            U_use = np.zeros((self.satellite.n_u, self.params.K))
            p_use = np.array([1.0])

        tf = float(p_use[0])
        ts = np.linspace(0.0, tf, self.params.K)

        cmds_list = []
        for k in range(self.params.K):
            u_k = U_use[:, k]
            cmds_list.append(SatelliteCommands(F_left=float(u_k[0]), F_right=float(u_k[1])))
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # states
        states_list = []
        for k in range(self.params.K):
            x_k = X_use[:, k]
            states_list.append(
                SatelliteState(
                    x=float(x_k[0]),
                    y=float(x_k[1]),
                    psi=float(x_k[2]),
                    vx=float(x_k[3]),
                    vy=float(x_k[4]),
                    dpsi=float(x_k[5]),
                )
            )
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states_list)

        return mycmds, mystates

    def _get_true_obstacle_cost(self, X_val: np.ndarray, p_val: float) -> float:
        """
        Calculates the actual nonlinear obstacle violation cost.
        Args:
            X_val: The state trajectory to evaluate
            p_val: The total flight time associated with X_val (CRITICAL for moving obstacles)
        """
        cost_obs = 0.0
        K = self.params.K

        # Gather obstacles
        all_obstacles = []
        if self.planets:
            all_obstacles.extend(self.planets.values())
        if self.asteroids:
            all_obstacles.extend(self.asteroids.values())

        r_sat = 1.4

        # --- FIX: Use the specific p_val passed to the function ---
        # This ensures we calculate asteroid positions based on the
        # trajectory's ACTUAL duration, not the previous iteration's guess.
        tf = float(p_val)
        dt = tf / (K - 1)

        for k in range(K):  # Check all points including last
            p_ship = X_val[0:2, k]

            # Calculate the specific time for this step based on p_val
            t_k = k * dt

            for obs in all_obstacles:
                # Determine obstacle position
                if hasattr(obs, "center"):  # Planet
                    p_obs = np.array(obs.center)
                    r_obs = float(obs.radius)
                elif hasattr(obs, "start") and hasattr(obs, "velocity"):  # Asteroid
                    start_pos = np.array(obs.start, dtype=float)

                    # velocity in LOCAL frame -> rotate to GLOBAL frame
                    vel_local = np.array(obs.velocity, dtype=float)
                    theta = float(getattr(obs, "orientation", 0.0))
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -s], [s, c]])
                    vel = R @ vel_local

                    # Project obstacle position to time t_k in global frame
                    p_obs = start_pos + vel * t_k
                    r_obs = float(obs.radius)
                else:
                    continue

                # Calculate Distance
                dist = np.linalg.norm(p_ship - p_obs)
                min_dist = r_sat + r_obs + self.params.safety_margin

                # If we are inside the margin, add penalty
                violation = max(0.0, min_dist - dist)
                cost_obs += violation

        # Apply the same weight as in the objective (1e5)
        return float(1e6 * cost_obs)

    def _visualize_guesses(self, IGs, start, goal):
        """Helper to plot and save generated initial guesses."""
        import matplotlib.pyplot as plt
        import os

        # Create a new figure
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # 1. Plot Planets
        if hasattr(self, "planets"):
            for p in self.planets.values():
                circle = plt.Circle(p.center, p.radius, color="gray", alpha=0.5)
                ax.add_patch(circle)
                # Label the planet
                ax.text(p.center[0], p.center[1], "Planet", ha="center", va="center")

        # 2. Plot Start/Goal
        ax.plot(start.x, start.y, "go", markersize=10, label="Start")
        ax.plot(goal.x, goal.y, "rx", markersize=10, label="Goal")

        # 3. Plot Trajectories
        colors = ["b", "g", "m", "c", "y"]
        for i, (X, _, _) in enumerate(IGs):
            c = colors[i % len(colors)]
            ax.plot(X[0, :], X[1, :], color=c, linewidth=2, linestyle="--", label=f"Guess {i}")

        ax.set_title("Generated Initial Guesses (Sandwich Topology)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        # 4. Save Logic
        # We save to "out_dir/13/" relative to where the script is run.
        try:
            # Create the directory if it doesn't exist
            output_dir = os.path.join(out_dir("13"), "index.html_resources", "IGs")
            os.makedirs(output_dir, exist_ok=True)

            # Save the file
            save_path = os.path.join(output_dir, f"{(np.abs(self.init_state.y)%1*100):.0f}.png")
            print("saving IGs")
            fig.savefig(save_path)
            print(f"[Planner] Initial guesses plot saved to: {save_path}")

        except Exception as e:
            print(f"[Planner] Could not save plot: {e}")

        # Close to free memory
        plt.close(fig)
