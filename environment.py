import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    WIDTH, HEIGHT, FPS, MAX_STEPS, HIDER_SPEED, SEEKER_SPEED,
    R_HIDER, R_SEEKER, FOV_HALF, FOV_RANGE, HEARING_RANGE, WALLS,
)
from geometry import norm, angle_diff, segment_hits_rect

class HideSeekEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)  # stay, up, down, left, right
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.hider = np.zeros(2, dtype=np.float32)
        self.seeker = np.zeros(2, dtype=np.float32)
        self.seeker_angle = 0.0
        self.seeker_goal = None
        self.last_seen = None
        self.last_seen_age = 0
        self.hider_last_seen = None
        self.hider_last_seen_age = 0
        self.search_steps = 0
        self.step_count = 0
        self.break_cooldown = 0
        self.was_spotted = False
        self.caught = False
        self.won = False

    def free(self, p, radius):
        x, y = float(p[0]), float(p[1])
        if x < radius or x > WIDTH - radius or y < radius or y > HEIGHT - radius:
            return False
        for rx, ry, rw, rh in WALLS:
            cx = np.clip(x, rx, rx + rw)
            cy = np.clip(y, ry, ry + rh)
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                return False
        return True

    def move(self, p, delta, radius):
        new = p.copy()
        for axis in range(2):
            test = new.copy()
            test[axis] += delta[axis]
            if self.free(test, radius):
                new = test
        return new

    def clear_los(self, a, b):
        return not any(segment_hits_rect(a, b, rect) for rect in WALLS)

    def seeker_spots_hider(self):
        vec = self.hider - self.seeker
        distance = float(np.linalg.norm(vec))
        if distance > FOV_RANGE:
            return False
        if not self.clear_los(self.seeker, self.hider):
            return False
        if distance < 37:
            return True  # cannot hide directly behind the seeker at touching range
        return abs(angle_diff(math.atan2(vec[1], vec[0]), self.seeker_angle)) <= FOV_HALF

    def hider_senses_seeker(self):
        d = float(np.linalg.norm(self.hider - self.seeker))
        return d <= HEARING_RANGE or (d < 310 and self.clear_los(self.hider, self.seeker))

    def random_free_point(self, radius):
        for _ in range(5000):
            p = self.rng.uniform([radius + 15, radius + 15],
                                 [WIDTH - radius - 15, HEIGHT - radius - 15]).astype(np.float32)
            if self.free(p, radius):
                return p
        raise RuntimeError("Could not sample an open spawn")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        self.seeker = self.random_free_point(R_SEEKER)
        for _ in range(5000):
            self.hider = self.random_free_point(R_HIDER)
            if np.linalg.norm(self.hider - self.seeker) > 280:
                break
        self.seeker_angle = float(self.rng.uniform(-math.pi, math.pi))
        self.seeker_goal = self.random_free_point(R_SEEKER)
        self.last_seen = None
        self.last_seen_age = 0
        self.hider_last_seen = None
        self.hider_last_seen_age = 0
        self.search_steps = 0
        self.step_count = 0
        self.break_cooldown = 0
        self.was_spotted = self.seeker_spots_hider()
        self.caught = False
        self.won = False
        return self.observation(), {}

    def clearance(self, direction):
        """Distance to first wall/boundary in a direction, normalized to 0..1."""
        max_dist = 100
        step = 5
        for dist in range(step, max_dist + step, step):
            if not self.free(self.hider + direction * dist, R_HIDER):
                return (dist - step) / max_dist
        return 1.0

    def observation(self):
        sensed = self.hider_senses_seeker()
        if sensed:
            self.hider_last_seen = self.seeker.copy()
            self.hider_last_seen_age = 0
        elif self.hider_last_seen is not None:
            self.hider_last_seen_age += 1
            if self.hider_last_seen_age > 25:
                self.hider_last_seen = None
        relative = ((self.seeker - self.hider) / [WIDTH, HEIGHT]) if sensed else [0., 0.]
        memory = ((self.hider_last_seen - self.hider) / [WIDTH, HEIGHT]
                  if self.hider_last_seen is not None else [0., 0.])
        clearances = [self.clearance(np.array(v, dtype=np.float32)) for v in
                      ((0, -1), (0, 1), (-1, 0), (1, 0))]
        obs = np.array([
            self.hider[0] / WIDTH, self.hider[1] / HEIGHT,
            *relative, float(sensed), float(self.seeker_spots_hider()),
            *clearances,
            *memory,
            float(self.hider_last_seen is not None),
            1.0 - self.step_count / MAX_STEPS,
            float(np.linalg.norm(self.seeker - self.hider) < 115),
        ], dtype=np.float32)
        return np.clip(obs, -1, 1)

    def seeker_step(self):
        visible = self.seeker_spots_hider()
        if visible:
            self.last_seen = self.hider.copy()
            self.last_seen_age = 0
            self.search_steps = 0
        elif self.last_seen is not None:
            self.last_seen_age += 1
            if self.last_seen_age > 50:
                self.last_seen = None

        if visible:
            goal = self.hider
        elif self.last_seen is not None:
            goal = self.last_seen
            if np.linalg.norm(goal - self.seeker) < 23:
                self.search_steps += 1
                if self.search_steps > 8:
                    self.last_seen = None
        else:
            goal = self.seeker_goal
            if np.linalg.norm(goal - self.seeker) < 30:
                self.seeker_goal = self.random_free_point(R_SEEKER)
                goal = self.seeker_goal

        d = norm(goal - self.seeker)
        if np.linalg.norm(d) < 1e-6:
            self.seeker_angle += 0.12
            return
        target_angle = math.atan2(d[1], d[0])
        rotation = np.clip(angle_diff(target_angle, self.seeker_angle), -0.38, 0.38)
        self.seeker_angle = angle_diff(self.seeker_angle + rotation, 0)
        heading = np.array([math.cos(self.seeker_angle), math.sin(self.seeker_angle)], dtype=np.float32)
        old = self.seeker.copy()
        self.seeker = self.move(self.seeker, heading * SEEKER_SPEED, R_SEEKER)
        if np.linalg.norm(self.seeker - old) < 1.0:
            # sidestep when a wall blocks direct pursuit
            for direction in (1.0, -1.0):
                tangent = np.array([-heading[1], heading[0]], dtype=np.float32) * direction
                moved = self.move(self.seeker, tangent * SEEKER_SPEED, R_SEEKER)
                if np.linalg.norm(moved - self.seeker) > 1.0:
                    self.seeker = moved
                    break
            if not visible and self.last_seen is None:
                self.seeker_goal = self.random_free_point(R_SEEKER)

    def step(self, action):
        directions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        direction = np.array(directions[action], dtype=np.float32)
        old = self.hider.copy()
        self.hider = self.move(self.hider, direction * HIDER_SPEED, R_HIDER)
        blocked = action != 0 and np.linalg.norm(self.hider - old) < 0.5
        self.seeker_step()
        self.step_count += 1
        spotted = self.seeker_spots_hider()
        distance = np.linalg.norm(self.hider - self.seeker)
        self.caught = distance <= R_HIDER + R_SEEKER
        time_up = self.step_count >= MAX_STEPS
        self.won = time_up and not self.caught
        terminated = self.caught or self.won
        reward = 0.015 + (0.025 if not spotted else 0.0)
        if self.was_spotted and not spotted and self.break_cooldown == 0:
            reward += 0.2
            self.break_cooldown = 20
        self.was_spotted = spotted
        self.break_cooldown = max(0, self.break_cooldown - 1)
        if blocked:
            reward -= 0.012
        if self.caught:
            reward -= 10.0
        if self.won:
            reward += 10.0
        info = {"survival_seconds": self.step_count / FPS,
                "caught": self.caught, "won": self.won, "visible": spotted}
        return self.observation(), float(reward), terminated, False, info
