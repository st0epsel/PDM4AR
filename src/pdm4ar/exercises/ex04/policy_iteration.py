import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, Cell, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:

        grid = grid_mdp.grid
        gamma = grid_mdp.gamma
        max_iter = 1001
        k_sweeps = max(grid.shape[0] + grid.shape[1], 20)
        print(f"k_sweeps: {k_sweeps}")

        # Policy and value_func setup
        value_func = np.zeros_like(grid).astype(float)
        policy = np.full_like(grid, 5)  # Action ABANDON
        for i, j in np.ndindex(grid.shape):
            if grid[i, j] == Cell.GOAL:
                policy[i, j] = 4  # Action STAY
            elif grid[i, j] == Cell.START:
                policy[i, j] = 0  # Action NORTH
        new_policy = np.copy(policy)

        for master_iter in range(max_iter):
            # === Policy Evaluation ===
            for policy_iter in range(k_sweeps):

                last_value_func = np.copy(value_func)
                for i, j in np.ndindex(value_func.shape):
                    state = (i, j)
                    if grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                        continue

                    action = Action(policy[state])
                    state_value = 0
                    all_action_outcomes = grid_mdp.get_transition_probability_reward(state)

                    if action in all_action_outcomes:
                        for next_state, prob, reward in all_action_outcomes[action]:
                            state_value += prob * (reward + gamma * last_value_func[next_state])

                    value_func[state] = state_value

            # === Policy Improvement ===
            policy_stable = True

            for i, j in np.ndindex(grid.shape):
                state = (i, j)
                if grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                    continue

                old_action = Action(policy[state])

                best_q_value = -np.inf
                best_action = old_action

                all_action_outcomes = grid_mdp.get_transition_probability_reward(state)

                for action, transitions in all_action_outcomes.items():
                    q_value = 0

                    for next_state, prob, reward in transitions:
                        q_value += prob * (reward + gamma * value_func[next_state])

                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = action

                new_policy[state] = int(best_action)

            if np.array_equal(policy, new_policy):
                policy = new_policy
                break

            policy = np.copy(new_policy)

        return value_func, policy


if __name__ == "__main__":
    mini_map = GridMdp(
        np.array(
            [
                [5, 4, 2, 3, 2, 2, 3, 2, 2, 2],
                [5, 2, 2, 3, 3, 2, 2, 5, 5, 2],
                [3, 5, 2, 2, 5, 2, 2, 2, 2, 2],
                [2, 3, 0, 2, 4, 2, 2, 2, 4, 5],
                [2, 2, 2, 2, 2, 3, 2, 3, 3, 2],
                [2, 2, 2, 5, 2, 2, 2, 2, 2, 3],
                [3, 5, 2, 2, 4, 2, 2, 2, 2, 3],
                [2, 2, 2, 3, 2, 2, 2, 1, 2, 2],
                [2, 2, 2, 3, 2, 2, 2, 2, 3, 3],
                [2, 5, 2, 3, 3, 2, 2, 3, 2, 2],
            ]
        )
    )
    print(PolicyIteration.solve(mini_map))
