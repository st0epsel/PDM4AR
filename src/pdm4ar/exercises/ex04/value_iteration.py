import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Action, Cell
from pdm4ar.exercises_def.ex04.utils import time_function


def print_dict(dictionary: dict):
    for key, item in dictionary.items():
        print(f"  action: {key}: {item}")
    print()


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        """
        print(f"grid: \n{str(grid_mdp.grid)}\n")
        grid_mdp.grid = grid_mdp.add_cliff_padding(grid_mdp.grid)
        print(f"grid: \n{str(grid_mdp.grid)}\n")
        grid_mdp.grid = grid_mdp.remove_cliff_padding(grid_mdp.grid)
        """
        grid = grid_mdp.grid
        last_value_func = np.zeros_like(grid).astype(float)
        value_func = np.copy(last_value_func)
        policy = np.zeros_like(grid).astype(int)

        gamma = grid_mdp.gamma

        tol = 0.0001
        max_reps = 10000
        value = 0

        for it in range(max_reps):
            delta = 0
            for i, j in np.ndindex(grid.shape):
                state = (i, j)
                if grid[state] != Cell.CLIFF and grid[state] != Cell.WONDERLAND:
                    max_value = -np.inf
                    transition_probability_reward = grid_mdp.get_transition_probability_reward(state)

                    for action, transitions in transition_probability_reward.items():
                        value = 0
                        for next_state, prob, reward in transitions:
                            value += prob * (reward + gamma * last_value_func[next_state])
                        max_value = max(max_value, value)
                    value_func[state] = max_value

            delta = np.max(np.abs(value_func - last_value_func))

            last_value_func = value_func.copy()

            value_func_out = np.copy(value_func)
            value_func_out[np.isnan(value_func_out)] = -1000.0

            # print(np.round(value_func))
            if delta < tol:
                break

        # Calculate Policy
        for i, j in np.ndindex(grid.shape):
            state = (i, j)
            if grid[state] == Cell.CLIFF or grid[state] == Cell.WONDERLAND:
                continue
            max_value = -np.inf
            max_value_action = Action.ABANDON
            transition_probability_reward = grid_mdp.get_transition_probability_reward(state)

            if grid[state] == Cell.GOAL:
                max_value_action = Action.STAY
                policy[state] = int(max_value_action)
                continue  # Go to the next state

            for action, transitions in transition_probability_reward.items():
                value = sum(
                    prob * (reward + gamma * value_func[next_state]) for next_state, prob, reward in transitions
                )
                # print(f"action: {action}, value: {value}")
                if value > max_value:
                    max_value = value
                    max_value_action = action
                # print(f"value: {value}, max_value: {max_value}")

            # print(f"max_value_action: {max_value_action}")
            policy[state] = int(max_value_action)

        """action_symbols = {
            Action.NORTH: "↑",
            Action.SOUTH: "↓",
            Action.EAST: "→",
            Action.WEST: "←",
            Action.STAY: "•",
            Action.ABANDON: "X",
        }
        symbolic_policy = np.vectorize(lambda a: action_symbols[Action(a)])(policy)
        """
        return (value_func, policy)


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
    print(ValueIteration.solve(mini_map))
