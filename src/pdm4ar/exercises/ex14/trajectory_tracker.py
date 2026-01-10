from os import close
from re import search
from typing import List, Tuple
import numpy as np
import math

from pdm4ar.exercises.ex14.global_config import GlobalConfig


class TrajectoryTracker:
    def __init__(self, trajectory: list[tuple[float, float]], lookahead_dist: float = GlobalConfig.FF_DIST):
        self.lookahead = lookahead_dist
        self.last_index = 0
        self.traj_finished = False
        self.valid = self.check_traj(trajectory)
        self.trajectory: np.ndarray
        self.new_traj(trajectory)
        self.actual_traj = []
        if self.valid:
            self.actual_traj.append(trajectory[0])

    def new_traj(self, trajectory):
        # print(f"   trajectory: \n{trajectory}")
        self.valid = self.check_traj(trajectory)
        self.traj_finished = False
        self.last_index = 0
        self.trajectory = trajectory

    def check_traj(self, trajectory) -> bool:
        return trajectory is not None and len(trajectory) >= 2

    def log_point(self, point):
        self.actual_traj.append(point)

    def get_actual_traj(self) -> List[Tuple[float, float]]:
        return self.actual_traj

    def get_trajectory(self) -> np.ndarray:
        return self.trajectory

    def get_lookahead_point_and_curvature(self, curr_x: float, curr_y: float) -> tuple[float, float, float]:
        if not self.valid:
            return curr_x, curr_y, 0.0  # Or return trajectory[0] if it exists

        if len(self.trajectory) < 3:
            # If length is 2, we can return the end point but cannot calc curvature.
            return self.trajectory[-1][0], self.trajectory[-1][1], 0.0

        # --- Step 1. Find the closest point to the robot ---
        closest_idx = self.last_index
        min_dist = float("inf")

        # Search window: Look at the next N points (e.g., 50)
        # We ensure search_limit does not exceed the list length
        search_limit = min(self.last_index + 50, len(self.trajectory))

        # (Restored logic you commented out)
        for i in range(self.last_index, search_limit):
            px, py = self.trajectory[i]
            d = math.hypot(px - curr_x, py - curr_y)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Update progress
        self.last_index = closest_idx

        # --- Step 2. Find the Lookahead Point ---
        target_idx = closest_idx
        found = False

        for i in range(closest_idx, len(self.trajectory)):
            px, py = self.trajectory[i]
            dist = math.hypot(px - curr_x, py - curr_y)
            if dist >= self.lookahead:
                target_idx = i
                found = True
                break

        if not found:
            # If we ran out of points, target is the very last point
            target_idx = len(self.trajectory) - 1
            self.traj_finished = True
        else:
            self.traj_finished = False

        # Safety clamp: Ensure target_idx is valid
        target_idx = min(target_idx, len(self.trajectory) - 1)
        tx, ty = self.trajectory[target_idx]

        # --- 3. Calculate Curvature (Menger) ---
        # We need (idx-1, idx, idx+1).
        # If target is Start (0) or End (len-1), we cannot compute curvature using neighbors.
        if target_idx <= 0 or target_idx >= len(self.trajectory) - 1:
            return tx, ty, 0.0

        p1 = self.trajectory[target_idx - 1]
        p2 = self.trajectory[target_idx]
        p3 = self.trajectory[target_idx + 1]

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
        d1 = math.hypot(x1 - x2, y1 - y2)
        d2 = math.hypot(x2 - x3, y2 - y3)
        d3 = math.hypot(x3 - x1, y3 - y1)

        if d1 * d2 * d3 == 0:
            k = 0.0
        else:
            k = 4 * area / (d1 * d2 * d3)

        cross_prod = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if cross_prod < 0:
            k = -k

        return tx, ty, k

    def reverse_point_and_curvature(self, curr_x: float, curr_y: float) -> tuple[float, float, float]:
        """
        Finds a target point backwards along the history (actual_traj).
        Stateless: Does not modify last_index or interrupt forward tracking.
        """
        # --- Validation ---
        if not self.valid or len(self.actual_traj) < 3:
            # Not enough history to calculate curvature
            # Return current pos and 0 curvature
            return curr_x, curr_y, 0.0

        # --- Find the closest point in HISTORY ---
        closest_back_idx = 1
        min_dist = float("inf")

        # Limit search to the last 50 points or total length
        search_limit = min(50, len(self.actual_traj))

        for i in range(1, search_limit):
            px, py = self.actual_traj[-i]
            d = math.hypot(px - curr_x, py - curr_y)

            if d < min_dist:
                min_dist = d
                closest_back_idx = i

        # --- Find the Lookahead Point ---
        target_back_idx = closest_back_idx
        found = False

        # Search deeper into history starting from the closest point
        for i in range(closest_back_idx, search_limit):
            px, py = self.actual_traj[-i]
            dist = math.hypot(px - curr_x, py - curr_y)

            if dist >= self.lookahead:
                target_back_idx = i
                found = True
                break

        # If we didn't find a point far enough back, clamp to the oldest point in search range
        if not found:
            target_back_idx = search_limit - 1

        max_idx = len(self.actual_traj) - 2
        target_back_idx = np.clip(target_back_idx, 2, max_idx)

        # Extract Target
        tx, ty = self.actual_traj[-target_back_idx]

        # --- Calculate Curvature  ---
        p1 = self.actual_traj[-(target_back_idx - 1)]  # Newer
        p2 = self.actual_traj[-target_back_idx]  # Current Target
        p3 = self.actual_traj[-(target_back_idx + 1)]  # Older

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
        d1 = math.hypot(x1 - x2, y1 - y2)
        d2 = math.hypot(x2 - x3, y2 - y3)
        d3 = math.hypot(x3 - x1, y3 - y1)

        if d1 * d2 * d3 == 0:
            k = 0.0
        else:
            k = 4 * area / (d1 * d2 * d3)

        cross_prod = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)

        if cross_prod < 0:
            k = -k

        return tx, ty, k
