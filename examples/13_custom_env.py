"""
Example 13: Custom Gymnasium Environment via BaseEnv
====================================================
Demonstrates how to implement a fully custom Gymnasium environment
by subclassing ``rlframework.envs.BaseEnv``:

- Inherit from ``BaseEnv`` — no boilerplate needed for ``reset/step`` signatures
- Only implement ``_reset()`` and ``_step()``
- Optional: override ``_render_rgb_array()`` / ``_render_human()``
- Register with ``tune.register_env()`` so RLlib workers can instantiate by name
- Train with CustomPPO

The env: **PointMass2D** — an agent navigates a 2D bounded box to reach a
random goal.  Observations are [agent_x, agent_y, goal_x, goal_y]; actions
are 2D velocity commands.

Run:
    python rlframework/examples/13_custom_env.py
"""

import gymnasium as gym
import numpy as np
import ray
from ray import tune

from rlframework.envs import BaseEnv
from rlframework.algorithms.ppo import CustomPPOConfig


# ==============================================================================
# 1. Define your custom env by subclassing BaseEnv
# ==============================================================================

class PointMass2D(BaseEnv):
    """2D point-mass navigation environment.

    Observation space: Box(-5, 5, shape=(4,))   → [agent_x, agent_y, goal_x, goal_y]
    Action space:      Box(-1, 1, shape=(2,))    → [vx, vy] velocity commands
    Reward:            -distance to goal, +10 on success
    Termination:       goal reached or max_episode_steps
    """

    def __init__(
        self,
        bounds: float = 5.0,
        max_speed: float = 0.5,
        success_threshold: float = 0.3,
    ):
        self.bounds = bounds
        self.max_speed = max_speed
        self.success_threshold = success_threshold

        obs_space = gym.spaces.Box(
            low=-bounds, high=bounds, shape=(4,), dtype=np.float32
        )
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        super().__init__(
            observation_space=obs_space,
            action_space=act_space,
            max_episode_steps=200,
        )

        self._agent_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._goal_pos: np.ndarray = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # BaseEnv abstract methods — only these two must be implemented
    # ------------------------------------------------------------------

    def _reset(self, seed, options):
        """Sample a new episode: agent at origin, goal randomly placed."""
        self._agent_pos = np.zeros(2, dtype=np.float32)

        if options and "goal" in options:
            self._goal_pos = np.asarray(options["goal"], dtype=np.float32)
        else:
            self._goal_pos = self.np_random.uniform(
                low=-self.bounds, high=self.bounds, size=2
            ).astype(np.float32)

        obs = np.concatenate([self._agent_pos, self._goal_pos])
        info = {"distance": self._distance(), "success": False}
        return obs, info

    def _step(self, action):
        """Apply velocity action, update position, compute reward."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        velocity = action * self.max_speed

        self._agent_pos = np.clip(
            self._agent_pos + velocity,
            -self.bounds,
            self.bounds,
        ).astype(np.float32)

        obs = np.concatenate([self._agent_pos, self._goal_pos])
        reward = self._compute_reward()
        terminated = self._is_success()
        info = {"distance": self._distance(), "success": terminated}

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _distance(self) -> float:
        return float(np.linalg.norm(self._agent_pos - self._goal_pos))

    def _is_success(self) -> bool:
        return self._distance() < self.success_threshold

    def _compute_reward(self) -> float:
        return 10.0 if self._is_success() else -self._distance()

    # ------------------------------------------------------------------
    # Rendering (optional)
    # ------------------------------------------------------------------

    def _render_rgb_array(self) -> np.ndarray:
        size = 400
        img = np.full((size, size, 3), 240, dtype=np.uint8)
        scale = size / (2 * self.bounds)
        cx, cy = size // 2, size // 2

        def to_px(pos):
            x = int(cx + pos[0] * scale)
            y = int(cy - pos[1] * scale)
            return x, y

        gx, gy = to_px(self._goal_pos)
        for dx in range(-60, 61):
            for dy in range(-60, 61):
                if dx * dx + dy * dy <= 3600:
                    px, py = gx + dx, gy + dy
                    if 0 <= px < size and 0 <= py < size:
                        img[py, px] = [40, 200, 80]

        ax, ay = to_px(self._agent_pos)
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                if dx * dx + dy * dy <= 100:
                    px, py = ax + dx, ay + dy
                    if 0 <= px < size and 0 <= py < size:
                        img[py, px] = [40, 80, 220]

        return img


# ==============================================================================
# 2. Register with tune so RLlib workers can find the env by name
# ==============================================================================

def make_pointmass_env(config=None):
    return PointMass2D(
        bounds=config.get("bounds", 5.0),
        max_speed=config.get("max_speed", 0.5),
        success_threshold=config.get("success_threshold", 0.3),
    )


tune.register_env("PointMass2D-v0", make_pointmass_env)

# Register with gymnasium so gym.make("PointMass2D-v0") works (sanity_check).
gym.register(id="PointMass2D-v0", entry_point=PointMass2D, max_episode_steps=200)


# ==============================================================================
# 3. Standalone sanity-check (no Ray)
# ==============================================================================

def sanity_check():
    """Random policy — verify env logic without RLlib."""
    print("=" * 60)
    print("Sanity-check: PointMass2D-v0 with random policy")
    print("=" * 60)

    env = gym.make("PointMass2D-v0", bounds=5.0, max_speed=0.5)
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    for ep in range(3):
        obs, _ = env.reset(seed=42 + ep)
        ep_ret = 0.0
        ep_len = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated

        print(
            f"  ep {ep}: return={ep_ret:8.2f}  "
            f"length={ep_len:4d}  "
            f"final_dist={info['distance']:.3f}  "
            f"success={info['success']}"
        )
    env.close()


# ==============================================================================
# 4. Train with CustomPPO
# ==============================================================================

def train():
    print("\n" + "=" * 60)
    print("Training: CustomPPO on PointMass2D-v0")
    print("=" * 60)

    ray.init(ignore_reinit_error=True)

    config = (
        CustomPPOConfig()
        .environment("PointMass2D-v0")
        .training(
            lr=3e-4,
            train_batch_size=2000,
            num_epochs=10,
            minibatch_size=256,
        )
        .env_runners(num_env_runners=2)
    )

    algo = config.build()

    for iteration in range(60):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        print(f"[iter {iteration:03d}] mean_reward={mean_reward:7.2f}")

    algo.stop()
    ray.shutdown()
    print("\nDone.")


# ==============================================================================
# 5. Entry point
# ==============================================================================

if __name__ == "__main__":
    sanity_check()
    train()
