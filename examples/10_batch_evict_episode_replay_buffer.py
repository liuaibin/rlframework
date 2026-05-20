"""
Example 10: Batch Evict Episode Replay Buffer
=============================================
Demonstrates:
- Testing BatchEvictEpisodeReplayBuffer against RLlib's EpisodeReplayBuffer
- Plugging BatchEvictEpisodeReplayBuffer into SAC via replay_buffer_config
- Inspecting local replay buffer stats during a short training run

Run:
    python examples/10_batch_evict_episode_replay_buffer.py

Fast behavior-only check without Ray training:
    BATCH_EVICT_TRAIN_ITERS=0 python examples/10_batch_evict_episode_replay_buffer.py
"""

import os

import ray
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

from rlframework.algorithms.sac import CustomSACConfig
from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

TOTAL_ITERATIONS = int(os.environ.get("BATCH_EVICT_TRAIN_ITERS", "10"))


def make_episode(
    episode_id: str,
    start: int,
    length: int,
    *,
    terminated: bool = True,
) -> SingleAgentEpisode:
    """Create a minimal SingleAgentEpisode for deterministic buffer checks."""
    return SingleAgentEpisode(
        id_=episode_id,
        observations=list(range(start, start + length + 1)),
        actions=list(range(start, start + length)),
        rewards=[1.0] * length,
        terminated=terminated,
        t_started=start,
        len_lookback_buffer=0,
    )


def verify_batch_evict_behavior() -> None:
    """Compare storage, indices, and metrics with the stock RLlib buffer."""
    base_buffer = EpisodeReplayBuffer(capacity=6)
    batch_evict_buffer = BatchEvictEpisodeReplayBuffer(capacity=6)

    add_inputs: list[SingleAgentEpisode | list[SingleAgentEpisode]] = [
        make_episode("A", 0, 2, terminated=False),
        make_episode("B", 0, 2),
        make_episode("A", 2, 2),
        [
            make_episode("C", 0, 2),
            make_episode("D", 0, 2),
            make_episode("E", 0, 2),
        ],
    ]

    for step, episodes in enumerate(add_inputs, start=1):
        base_buffer.add(episodes)
        batch_evict_buffer.add(episodes)

        assert base_buffer.get_state() == batch_evict_buffer.get_state()
        assert base_buffer.get_metrics() == batch_evict_buffer.get_metrics()
        assert base_buffer.get_num_timesteps() == batch_evict_buffer.get_num_timesteps()
        assert base_buffer.get_num_episodes() == batch_evict_buffer.get_num_episodes()
        assert (
            base_buffer.get_num_episodes_evicted() == batch_evict_buffer.get_num_episodes_evicted()
        )

        print(
            f"[check {step}] "
            f"episodes={batch_evict_buffer.get_num_episodes()}  "
            f"timesteps={batch_evict_buffer.get_num_timesteps()}  "
            f"evicted={batch_evict_buffer.get_num_episodes_evicted()}  "
            f"indices={len(batch_evict_buffer._indices)}"
        )

    assert batch_evict_buffer.episode_id_to_index == {"C": 2, "D": 3, "E": 4}
    assert batch_evict_buffer._indices == [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
    ]
    print("[verified] BatchEvictEpisodeReplayBuffer matches EpisodeReplayBuffer.")


def build_config() -> CustomSACConfig:
    """Build a SAC config that uses the batch-evict replay buffer."""
    return (
        CustomSACConfig()
        .framework_run("batch_evict_replay_buffer", root_dir="./runs")
        .environment("Pendulum-v1")
        .training(
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            train_batch_size_per_learner=256,
            replay_buffer_config={
                "type": BatchEvictEpisodeReplayBuffer,
                "capacity": 50_000,
            },
            target_entropy="auto",
            tau=0.005,
        )
        .env_runners(num_env_runners=1, rollout_fragment_length=1)
        .metrics(reporters=["file"])
    )


def train_with_batch_evict_buffer(total_iterations: int = TOTAL_ITERATIONS) -> None:
    """Run a short SAC training loop and verify the configured buffer type."""
    if total_iterations <= 0:
        print("[skip] BATCH_EVICT_TRAIN_ITERS=0, skipped SAC training.")
        return

    ray.init(ignore_reinit_error=True)
    config = build_config()
    algo = config.build()

    try:
        for iteration in range(total_iterations):
            result = algo.train()
            mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))

            replay_buffer = algo.local_replay_buffer
            assert isinstance(replay_buffer, BatchEvictEpisodeReplayBuffer), (
                f"Expected BatchEvictEpisodeReplayBuffer, got {type(replay_buffer)}"
            )

            print(
                f"[iter {iteration:03d}] "
                f"reward={mean_reward:.2f}  "
                f"buf_size={len(replay_buffer)}  "
                f"episodes={replay_buffer.get_num_episodes()}  "
                f"added={replay_buffer.get_added_timesteps()}  "
                f"sampled={replay_buffer.get_sampled_timesteps()}"
            )
    finally:
        algo.stop()
        ray.shutdown()

    print("\nDone. SAC trained with BatchEvictEpisodeReplayBuffer.")


if __name__ == "__main__":
    verify_batch_evict_behavior()
    train_with_batch_evict_buffer()
