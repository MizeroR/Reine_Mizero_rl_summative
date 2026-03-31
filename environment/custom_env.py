"""
Digital Wellbeing Coach — Smartphone Addiction Risk & Intervention Recommender
Custom Gymnasium Environment

Observation Space (7 continuous features):
    [0] screen_time      : daily screen time in hours          [0, 16]
    [1] unlocks          : number of phone unlocks per day     [0, 200]
    [2] sleep_quality    : sleep quality score                 [0, 10]
    [3] sas_score        : SAS-SV addiction risk score         [0, 48]
    [4] social_ratio     : proportion of time on social media  [0, 1]
    [5] productivity_ratio: proportion of time on productivity [0, 1]
    [6] time_of_day      : current hour                        [0, 23]

Action Space (6 discrete actions):
    0 — Do nothing
    1 — Reduce screen time by 30 min
    2 — Block social media
    3 — Send mindfulness prompt
    4 — Enforce a break
    5 — Enable focus mode

Terminal Conditions:
    - SAS-SV score < 24 (user reaches mild-risk status)
    - Episode reaches 30 timesteps
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Compliance probabilities per action (how likely the user follows the intervention)
ACTION_COMPLIANCE = {
    0: 1.00,   # do nothing — always "complied" (no action taken)
    1: 0.65,   # reduce screen time
    2: 0.55,   # block social media
    3: 0.75,   # mindfulness prompt (low friction)
    4: 0.50,   # enforce a break (moderate resistance)
    5: 0.70,   # focus mode
}

# Base SAS-SV delta when an intervention is complied with
ACTION_SAS_DELTA = {
    0:  0.0,
    1: -1.8,
    2: -2.2,
    3: -1.2,
    4: -1.5,
    5: -1.6,
}

ACTION_NAMES = [
    "Do Nothing",
    "Reduce Screen Time (-30 min)",
    "Block Social Media",
    "Send Mindfulness Prompt",
    "Enforce a Break",
    "Enable Focus Mode",
]


class WellbeingEnv(gym.Env):
    """
    Custom Gymnasium environment modelling a simulated user's smartphone
    addiction risk over time, with an agent recommending interventions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = 30

        # ── Observation space ──────────────────────────────────────────────
        low  = np.array([0.0,   0.0, 0.0,  0.0, 0.0, 0.0,  0.0], dtype=np.float32)
        high = np.array([16.0, 200.0, 10.0, 48.0, 1.0, 1.0, 23.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action space ───────────────────────────────────────────────────
        self.action_space = spaces.Discrete(6)

        # Internal state (populated in reset)
        self.state: np.ndarray = np.zeros(7, dtype=np.float32)
        self.step_count: int = 0
        self.last_action: int = 0
        self.last_complied: bool = False
        self.history: list[dict] = []

        # Renderer (lazy-initialised)
        self._renderer = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        return self.state.copy()

    def _random_high_risk_state(self, rng: np.random.Generator) -> np.ndarray:
        """Generate an initial high-risk user state (SAS-SV > 30)."""
        sas      = float(rng.uniform(30, 48))
        screen   = float(rng.uniform(6, 12))           # heavy usage
        unlocks  = float(rng.uniform(80, 180))
        sleep    = float(rng.uniform(2, 6))             # poor sleep
        social   = float(rng.uniform(0.4, 0.8))
        prod     = float(rng.uniform(0.05, 0.2))
        tod      = float(rng.integers(0, 24))
        return np.array([screen, unlocks, sleep, sas, social, prod, tod],
                        dtype=np.float32)

    # ── Core API ───────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.state = self._random_high_risk_state(rng)
        self.step_count = 0
        self.last_action = 0
        self.last_complied = False
        self.history = []
        obs = self._obs()
        info = {"sas_score": float(self.state[3]), "step": 0}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_sas = float(self.state[3])

        # ── Determine compliance ───────────────────────────────────────────
        complied = (action == 0) or (
            self.np_random.random() < ACTION_COMPLIANCE[action]
        )
        self.last_complied = complied
        self.last_action   = action

        # ── Update SAS-SV ──────────────────────────────────────────────────
        delta_sas = ACTION_SAS_DELTA[action] if complied else 0.0
        # Natural drift: risk tends to increase slightly without intervention
        natural_drift = float(self.np_random.uniform(0.1, 0.4))
        new_sas = float(np.clip(prev_sas + delta_sas + natural_drift, 0, 48))

        # ── Update screen time ─────────────────────────────────────────────
        screen = float(self.state[0])
        if action == 1 and complied:
            screen = max(0.0, screen - 0.5)
        elif action in (2, 4, 5) and complied:
            screen = max(0.0, screen - self.np_random.uniform(0.2, 0.6))
        else:
            screen = float(np.clip(screen + self.np_random.uniform(-0.3, 0.6), 0, 16))

        # ── Update unlocks ─────────────────────────────────────────────────
        unlocks = float(self.state[1])
        if action in (2, 4, 5) and complied:
            unlocks = max(0.0, unlocks - self.np_random.uniform(5, 15))
        else:
            unlocks = float(np.clip(unlocks + self.np_random.uniform(-5, 10), 0, 200))

        # ── Update sleep quality ───────────────────────────────────────────
        sleep = float(self.state[2])
        if action == 3 and complied:           # mindfulness improves sleep
            sleep = min(10.0, sleep + self.np_random.uniform(0.2, 0.8))
        elif action == 4 and complied:         # break helps too
            sleep = min(10.0, sleep + self.np_random.uniform(0.1, 0.4))
        else:
            sleep = float(np.clip(sleep + self.np_random.uniform(-0.3, 0.3), 0, 10))

        # ── Update app-category ratios ────────────────────────────────────
        social = float(self.state[4])
        prod   = float(self.state[5])
        if action == 2 and complied:
            social = max(0.0, social - self.np_random.uniform(0.05, 0.15))
            prod   = min(1.0, prod   + self.np_random.uniform(0.02, 0.08))
        elif action == 5 and complied:
            social = max(0.0, social - self.np_random.uniform(0.02, 0.08))
            prod   = min(1.0, prod   + self.np_random.uniform(0.05, 0.12))
        else:
            social = float(np.clip(social + self.np_random.uniform(-0.03, 0.05), 0, 1))
            prod   = float(np.clip(prod   + self.np_random.uniform(-0.02, 0.03), 0, 1))

        # Keep ratios from exceeding 1 together
        total_ratio = social + prod
        if total_ratio > 1.0:
            scale = 1.0 / total_ratio
            social *= scale
            prod   *= scale

        # ── Advance time of day ────────────────────────────────────────────
        tod = float((int(self.state[6]) + 1) % 24)

        # ── Compose new state ──────────────────────────────────────────────
        self.state = np.array(
            [screen, unlocks, sleep, new_sas, social, prod, tod],
            dtype=np.float32
        )
        self.step_count += 1

        # ── Compute reward ─────────────────────────────────────────────────
        sas_improvement = prev_sas - new_sas          # positive = improvement
        reward = 0.0

        if sas_improvement > 0:
            reward += sas_improvement * 1.5           # reward risk reduction
        else:
            reward += sas_improvement * 1.0           # penalise risk increase

        if complied and action != 0:
            reward += 1.0                             # compliance bonus
        elif not complied and action != 0:
            reward -= 0.5                             # non-compliance penalty

        # Bonus for reaching mild-risk status
        if new_sas < 24:
            reward += 10.0

        # Small penalty for aggressive interventions when risk is already low
        if new_sas < 24 and action in (2, 4):
            reward -= 0.3

        # ── Terminal check ─────────────────────────────────────────────────
        terminated = bool(new_sas < 24)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "sas_score":   new_sas,
            "prev_sas":    prev_sas,
            "complied":    complied,
            "action_name": ACTION_NAMES[action],
            "step":        self.step_count,
        }

        self.history.append({
            "step":    self.step_count,
            "action":  action,
            "sas":     new_sas,
            "reward":  reward,
            "complied": complied,
        })

        if self.render_mode == "human":
            self._render_frame()

        return self.state.copy(), reward, terminated, truncated, info

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        from environment.rendering import WellbeingRenderer
        if self._renderer is None:
            self._renderer = WellbeingRenderer()
        return self._renderer.draw(self.state, self.last_action, self.last_complied,
                                   self.step_count, self.history)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Convenience ────────────────────────────────────────────────────────

    def get_state_dict(self) -> dict:
        """Return current state as a labelled dictionary."""
        s = self.state
        return {
            "screen_time":         round(float(s[0]), 2),
            "unlocks":             int(s[1]),
            "sleep_quality":       round(float(s[2]), 2),
            "sas_score":           round(float(s[3]), 2),
            "social_ratio":        round(float(s[4]), 3),
            "productivity_ratio":  round(float(s[5]), 3),
            "time_of_day":         int(s[6]),
        }
