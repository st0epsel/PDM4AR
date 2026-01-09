import math as m
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AgentState:
    def __init__(self, radius: float = 0.5):
        self.x: float = 0.0
        self.y: float = 0.0
        self.psi: float = 0.0
        self.radius: float = radius
        self.time: float = 0.0
        self.heading: float = 0.0

        # Priority (default 0, optional update)
        self.priority: float = 0.0

        # Linear velocity (v) and Angular velocity (dpsi)
        # v is scalar linear velocity (signed). None if not yet observed.
        self.v: Optional[float] = None
        self.dpsi: Optional[float] = None

        # Direction: 1.0 for Forward, -1.0 for Backward
        # Defaults to Forward (1.0) until observed otherwise
        self.direction: float = 1.0

        # Private history for finite differencing
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._last_psi: Optional[float] = None

    def __call__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {(self.heading/np.pi*180):.2f})"

    def update(self, state_vector: np.ndarray, sim_time: float, priority: Optional[float] = None):
        """
        Updates state, estimates velocities, tracks direction, and optionally updates priority.

        :param state_vector: np.array([x, y, psi])
        :param sim_time: The current simulation time
        :param priority: (Optional) New priority value
        """
        x, y, psi = float(state_vector[0]), float(state_vector[1]), float(state_vector[2])
        dt = sim_time - self.time

        # 0. Update Priority if provided
        if priority is not None:
            self.priority = priority

        # 1. Estimate Velocities & Direction
        if self._last_x is not None and dt > 1e-4:
            # Distance moved
            dx = x - self._last_x
            dy = y - self._last_y
            dist = np.hypot(dx, dy)
            speed = dist / dt

            # Determine direction (Forward vs Backward)
            # Project displacement onto the robot's current heading vector
            longitudinal_displacement = dx * np.cos(psi) + dy * np.sin(psi)

            # Only update direction if movement is significant (avoids noise at standstill)
            if abs(longitudinal_displacement) > 1e-5:
                self.direction = np.sign(longitudinal_displacement)
                if self.direction == 0:
                    self.direction = 1.0

            self.heading = psi + (np.pi / 2) * self.direction - np.pi / 2

            # Linear Velocity is Speed * Direction
            self.v = speed * self.direction

            # Angular Velocity Calculation
            # Handle angle wrapping [-pi, pi]
            delta_psi = psi - self._last_psi
            delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi
            self.dpsi = delta_psi / dt
        else:
            self.v = None
            self.dpsi = None
            self.heading = psi
            # Note: We do NOT reset self.direction to None;
            # we keep the last known gear (forward/reverse).

        # 2. Update State
        self.x = x
        self.y = y
        self.psi = psi
        self.time = sim_time

        # 3. Store History
        self._last_x = self.x
        self._last_y = self.y
        self._last_psi = self.psi

    def get_distance(self, other: "AgentState") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return m.hypot(dx, dy)

    def get_collision_dist(
        self, other: "AgentState", time_horizon: float = 5.0, step_size: float = 0.2
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Predicts time to collision by simulating both robots forward along circular arcs.
        """
        # Safety: Need velocities to predict
        if self.v is None or other.v is None:
            return None

        # Default dpsi to 0.0 if unobserved
        w1 = self.dpsi if self.dpsi is not None else 0.0
        w2 = other.dpsi if other.dpsi is not None else 0.0

        v1 = self.v
        v2 = other.v

        # Initialize simulation states
        x1, y1, th1 = self.x, self.y, self.psi
        x2, y2, th2 = other.x, other.y, other.psi
        self_r, other_r = self.radius, other.radius

        # Collision distance squared
        min_dist_sq = (self_r + other_r) ** 2

        # Simulation Loop
        t = 0.0
        while t < time_horizon:
            # Check for collision
            dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2

            if dist_sq < min_dist_sq * (1 + t / (2 * time_horizon)):
                return t, x2, y2, self.get_distance(other) - (self_r + other_r)

            # Update Robot 1
            x1 += v1 * np.cos(th1) * step_size
            y1 += v1 * np.sin(th1) * step_size
            th1 += w1 * step_size

            # Update Robot 2
            x2 += v2 * np.cos(th2) * step_size
            y2 += v2 * np.sin(th2) * step_size
            th2 += w2 * step_size

            t += step_size

        return None

    def is_in_front_of(self, other: "AgentState") -> bool:
        """
        Returns True if 'self' is physically behind 'other'
        (based on 'other's' current position and heading).
        """
        # 1. Vector from 'Other' (Origin) to 'Self' (Target)
        dx = self.x - other.x
        dy = self.y - other.y

        if other.heading is None:
            return False

        # 2. Project onto 'Other's' heading (FIXED: was using self.psi)
        # Dot Product = |A||B|cos(theta)
        longitudinal_rel = dx * np.cos(other.heading) + dy * np.sin(other.heading)

        # 3. If negative, I am "behind" the plane perpendicular to his nose.
        return longitudinal_rel < 0
