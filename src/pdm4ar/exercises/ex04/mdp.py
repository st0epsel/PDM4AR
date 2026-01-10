from abc import ABC, abstractmethod
from http.client import GONE
from operator import is_
from shutil import move

import numpy as np
from numpy.typing import NDArray
from traitlets import Bool
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc, Cell


def add(a: State, b: State) -> State:
    return (a[0] + b[0], a[1] + b[1])


class ActionVector:
    """Helper class to represent the effect of each action as a vector"""

    VECTORS = {
        Action.NORTH: (-1, 0),
        Action.WEST: (0, -1),
        Action.SOUTH: (1, 0),
        Action.EAST: (0, 1),
        Action.STAY: (0, 0),
        Action.ABANDON: (0, 0),
    }

    @staticmethod
    def get_vector(action: Action) -> tuple[int, int]:
        return ActionVector.VECTORS[action]


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        grid_shape = grid.shape
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""
        self.start_cell = self.find_start()
        self.padded = False

        self.admissible_actions = np.full((grid_shape[0], grid_shape[1], 5), np.nan)
        self.gridcalc_admissible_actions()

        self.possible_next_states = np.full((grid_shape[0], grid_shape[1], 10, 2), np.nan)
        self.gridcalc_possible_next_states()

        self.transition_prob_rews = np.full((grid_shape[0], grid_shape[1], 5, 9, 2), np.nan)
        self.gridcalc_transition_probability_reward()

        # 1. Add a cache for the final dictionaries
        self.prob_reward_dict = {}

        # 2. Add a call to a new function that will populate this cache
        self.build_prob_reward_dict()

    def build_prob_reward_dict(self):
        """
        Builds the probability reward dictionary
        """
        for i, j in np.ndindex(self.grid.shape):
            state = (i, j)
            # Skip states that are never 'decision' states
            if self.grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                self.prob_reward_dict[state] = {}
                continue

            # This is the logic from your get_transition_probability_reward
            transition_prob_rews = self.transition_prob_rews
            transition_probability_reward = {}

            # Pre-calculate this once per state
            next_states = tuple(set(self.get_possible_next_states(state)))

            for action_nr, action in enumerate(self.get_admissible_actions(state)):
                next_state_action = []

                # Special hardcoded case (this is good)
                if action == Action.ABANDON:
                    transition_probability_reward[Action.ABANDON] = tuple([(self.start_cell, 1.0, -10.0)])
                    continue

                # Build the (next_state, prob, reward) tuples
                for next_state_nr, next_state in enumerate(next_states):
                    next_state_action_probability = self.transition_prob_rews[
                        state[0], state[1], action_nr, next_state_nr, 0
                    ]
                    if not np.isnan(next_state_action_probability):
                        next_state_action.append(
                            (
                                next_state,
                                next_state_action_probability,
                                transition_prob_rews[state[0], state[1], action_nr, next_state_nr, 1],
                            )
                        )
                if len(next_state_action) > 0:
                    transition_probability_reward[action] = tuple(next_state_action)

            # Save the final, expensive-to-build dictionary to the cache
            self.prob_reward_dict[state] = transition_probability_reward

    def get_transition_probability_reward(self, state: State) -> dict[Action, tuple[State, float, float]]:
        return self.prob_reward_dict.get(state, {})

    def find_start(self) -> tuple[int, int]:
        grid = self.grid
        for i, j in np.ndindex(self.grid.shape):
            if grid[i, j] == Cell.START:
                return (i, j)
        assert False, "No Start found in grid"

    def check_legal_state(self, state: State) -> bool:
        rows, cols = self.grid.shape
        return 0 <= state[0] < rows and 0 <= state[1] < cols and self.grid[state] != Cell.CLIFF

    def calc_admissible_actions(self, state: State) -> tuple:
        grid_state = self.grid[state]
        if grid_state == Cell.GOAL:
            return (Action.STAY,)

        action_list = []

        if not self.start_cell_or_direct_neighbor(state):
            action_list.append(Action.ABANDON)

        for action in [Action.WEST, Action.NORTH, Action.EAST, Action.SOUTH]:
            planned_state = add(ActionVector.get_vector(action), state)
            if self.check_legal_state(planned_state):
                action_list.append(action)

        return tuple(action_list)

    def gridcalc_admissible_actions(self):
        admissible_actions = self.admissible_actions
        grid = self.grid
        for i, j in np.ndindex(admissible_actions.shape[:2]):
            if grid[(i, j)] != Cell.WONDERLAND and grid[(i, j)] != Cell.CLIFF:
                actions = self.calc_admissible_actions((i, j))
                max_size = 5  # Only ever Action.STAY OR Action.ABANDON, never both
                padded = list(actions[:5]) + [np.nan] * (max_size - len(actions))
                admissible_actions[i, j, :] = padded

    def get_admissible_actions(self, state: State) -> tuple[Action, ...]:
        actions = self.admissible_actions[state]
        actions = actions[~np.isnan(actions)]
        return tuple(actions.astype(Action))

    def get_outcomes(self, state: State, action: Action) -> dict[State, float]:
        """
        Returns a dictionary mapping {next_state: probability}
        for all possible next_states.
        """
        outcomes = {}  # Use a dict to automatically sum probabilities
        grid_state = self.grid[state]
        start_cell = self.start_cell

        # Processing a single move
        def add_move_outcome(move_action: Action, move_prob: float):
            target_state = add(state, ActionVector.get_vector(move_action))

            # === Move is off-grid or to a CLIFF
            if not self.check_legal_state(target_state):
                outcomes[start_cell] = outcomes.get(start_cell, 0.0) + move_prob
                return

            target_cell_type = self.grid[target_state]

            # === Move is to WONDERLAND
            if target_cell_type == Cell.WONDERLAND:
                teleport_options = []
                for vec in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dest = add(target_state, vec)
                    if dest != state:  # Cannot teleport back to where you were
                        teleport_options.append(dest)

                if not teleport_options:  # Should not happen based on problem
                    outcomes[start_cell] = outcomes.get(start_cell, 0.0) + move_prob
                    return

                teleport_prob = move_prob / len(teleport_options)
                for dest in teleport_options:
                    if not self.check_legal_state(dest):
                        outcomes[start_cell] = outcomes.get(start_cell, 0.0) + teleport_prob
                    else:
                        outcomes[dest] = outcomes.get(dest, 0.0) + teleport_prob

            # === Normal, valid move
            else:
                outcomes[target_state] = outcomes.get(target_state, 0.0) + move_prob

        # === Main Logic
        if grid_state == Cell.GOAL:
            if action == Action.STAY:
                outcomes[state] = 1.0

        elif action == Action.ABANDON:
            outcomes[start_cell] = 1.0

        elif grid_state in (Cell.GRASS, Cell.START):
            all_moves = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
            other_moves = [m for m in all_moves if m != action]

            # Intended move
            add_move_outcome(action, 0.75)
            # Other moves
            for move in other_moves:
                add_move_outcome(move, 0.25 / 3.0)

        elif grid_state == Cell.SWAMP:
            outcomes[start_cell] = outcomes.get(start_cell, 0.0) + 0.05
            outcomes[state] = outcomes.get(state, 0.0) + 0.20

            all_moves = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
            other_moves = [m for m in all_moves if m != action]
            add_move_outcome(action, 0.5)

            for move in other_moves:
                add_move_outcome(move, 0.25 / 3.0)

        return outcomes

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        all_outcomes = self.get_outcomes(state, action)
        return all_outcomes.get(next_state, 0.0)

    def calc_possible_next_states(self, state: State) -> tuple[State]:
        grid = self.grid
        grid_state = grid[state]

        if grid_state == Cell.GOAL:
            return (state,)

        states = [self.start_cell]  # If action Abandon

        if grid_state == Cell.SWAMP:
            transitions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        else:
            transitions = ((1, 0), (0, 1), (-1, 0), (0, -1))

        for transition in transitions:
            new_state = add(state, transition)
            if not self.check_legal_state(new_state):
                states.append(self.start_cell)
            elif grid[new_state] in (Cell.START, Cell.GRASS, Cell.SWAMP, Cell.GOAL):
                states.append(new_state)

            elif grid[new_state] == Cell.WONDERLAND:
                wonderland_transitions = ((1, 0), (0, 1), (-1, 0), (0, -1))
                for wonderland_transition in wonderland_transitions:
                    new_wonderland_state = add(new_state, wonderland_transition)
                    if not self.check_legal_state(new_wonderland_state):
                        states.append(self.start_cell)
                    elif new_wonderland_state != state:
                        states.append(new_wonderland_state)
        # print(f"state: {state}, possible_states: {states}")
        return tuple(states)

    def gridcalc_possible_next_states(self):
        possible_next_states = self.possible_next_states
        grid = self.grid
        for i, j in np.ndindex(possible_next_states.shape[:2]):
            if grid[(i, j)] != Cell.WONDERLAND and grid[(i, j)] != Cell.CLIFF:
                next_states = np.array(self.calc_possible_next_states((i, j)), dtype=float)
                max_size = (10, 2)  # Worst-case-scenario: current state: Cell.SWAMP, two adjacent Cell.WONDERLAND
                padded = np.full(max_size, np.nan)
                padded[: next_states.shape[0], :] = next_states
                possible_next_states[i, j:, :, :] = padded[:, :]

    def get_possible_next_states(self, state: State) -> tuple[State, ...]:
        states = self.possible_next_states[state]
        valid_states = states[~np.isnan(states).any(axis=1)]
        return tuple(map(tuple, valid_states.astype(int)))

    def is_adjacient(self, state1: State, state2: State) -> bool:
        return (state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2 <= 1.0

    def get_adjacient_wonderlands(self, state: State) -> tuple:
        wonderlands = []
        for transition in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            cell = add(state, transition)
            if self.check_legal_state(cell) and Cell.WONDERLAND == self.grid[cell]:
                wonderlands.append(cell)
        return tuple(wonderlands)

    def start_cell_or_direct_neighbor(self, state) -> bool:
        for check_vector in ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)):
            if add(state, check_vector) == self.start_cell:
                return True
        return False

    def get_n_adjacent_prohibited_cells(self, state: State) -> int:
        # Precompute??
        n_cliffs = 0
        for check_vector in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            check_state = add(state, check_vector)
            if not self.check_legal_state(check_state):
                n_cliffs += 1
        return n_cliffs

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        grid_state = self.grid[state]

        if grid_state == Cell.GOAL:
            if action == Action.STAY:
                return 50.0
            return 0.0

        reward = 0.0

        # If we don't abandon the bot, the cost ist dictated by the grid field and the time we spend in it
        if next_state == self.start_cell:
            if action == Action.ABANDON:
                reward -= 10.0
            elif not self.start_cell_or_direct_neighbor(state):
                reward -= 10.0

        # Time Cost
        if action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
            if grid_state in (Cell.GRASS, Cell.START):
                reward -= 1.0
            elif grid_state == Cell.SWAMP:
                reward -= 2.0

        if action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
            planned_state = add(state, ActionVector.get_vector(action))
            if self.check_legal_state(planned_state) and self.grid[planned_state] == Cell.WONDERLAND:
                if next_state != self.start_cell:
                    reward += 3.0

        return reward

    def calc_transition_prob_rewards(self, state: State):
        transition_probability_reward = {}
        non_normal_psum = False
        psum = {}
        for action in self.get_admissible_actions(state):
            next_state_action = []
            action_probability = 0
            # print(f"possible next states: {self.get_possible_next_states(state)}")
            # Remove duplicate next_state == Cell.START
            for next_state in tuple(set(self.get_possible_next_states(state))):
                next_state_action_probability = self.get_transition_prob(state, action, next_state)
                action_probability += next_state_action_probability
                if next_state_action_probability > 0.0:
                    next_state_action.append(
                        (
                            next_state,
                            next_state_action_probability,
                            self.stage_reward(state, action, next_state),
                        )
                    )
            # print(f"total action ({action}) action_probability: {action_probability}")
            if abs(action_probability - 1.0) > 1e-6:
                non_normal_psum = True
            transition_probability_reward[action] = tuple(next_state_action)
            psum[action] = action_probability
        # print(f"transition_probability_reward: {transition_probability_reward}")

        if non_normal_psum:
            print(f"state: {state},    admissible actions: {self.get_admissible_actions(state)}")
            print_dict(transition_probability_reward, psum)

        return transition_probability_reward

    def gridcalc_transition_probability_reward(self):
        # print("gridcalc_transition_probability... ")
        transition_prob_rews = self.transition_prob_rews
        grid = self.grid
        for i, j in np.ndindex(transition_prob_rews.shape[:2]):
            if grid[(i, j)] != Cell.WONDERLAND and grid[(i, j)] != Cell.CLIFF:
                state = (i, j)
                for action_nr, action in enumerate(self.get_admissible_actions(state)):
                    action_probability = 0
                    outstring = ""
                    next_states = tuple(set(self.get_possible_next_states(state)))
                    for next_state_nr, next_state in enumerate(next_states):
                        next_state_action_probability = self.get_transition_prob(state, action, next_state)
                        next_state_action_reward = self.stage_reward(state, action, next_state)
                        if next_state_action_probability > 0.0:
                            transition_prob_rews[i, j, action_nr, next_state_nr, 0:2] = np.array(
                                [next_state_action_probability, next_state_action_reward]
                            )
                            action_probability += next_state_action_probability
                        outstring += f"p({next_state}) = {np.round(next_state_action_probability,4)}, r = {np.round(next_state_action_reward,4)} "

                    if abs(action_probability - 1.0) >= 0.001:
                        print(f"\n{action_probability} = total action probatility ")
                        print(f"state: {state}, action: {action}")
                        print(f"outstring: {outstring}")
                        print(f"possible_next_states: {next_states}")
                        print(f"get_transition_prob_rew from memory: ")
                        print_dict1(self.get_transition_probability_reward(state))

        self.transition_prob_rews = transition_prob_rews


def print_dict(dictionary: dict, psum: dict):
    for key, item in dictionary.items():
        print(f"  action: {key} (psum: {psum[key]}): {item}")
    print()


def print_dict1(dictionary: dict):
    for key, item in dictionary.items():
        print(f"  action: {key}: {item}")
    print()


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        grid = grid_mdp.grid
        value_func = np.zeros_like(grid).astype(float)
        policy = np.zeros_like(grid).astype(int)  # Contains the best action in any field??

        return (value_func, policy)


if __name__ == "__main__":
    MiniMap = GridMdp(np.array([[5, 2, 2, 2, 5], [4, 3, 2, 3, 4], [2, 2, 1, 2, 0], [4, 3, 2, 3, 4], [5, 2, 2, 2, 5]]))
    """print(MiniMap.grid)
    states = ((1, 1), (2, 0))

    for state in states:
        print(f"MiniMap.get_possible_next_states{state} = {MiniMap.get_possible_next_states(state)}")
        print(MiniMap.get_possible_next_states(state))"""
