from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque
import argparse
import random
import numpy as np
import jax
import jax.numpy as jnp

from minatar_env import MinatarEnv
from network import MuZeroNet
from mcts import MuZeroMCTS, MCTSconfig
from buffer import Episode, EpisodeBuffer, StepData
from trainer import MuZeroTrainer, TrainConfig


@dataclass(frozen=True)
class RunConfig:
    env_name: str = "breakout"
    seed: int = 42

    num_episodes: int = 200
    max_steps_per_episode: int = 500

    train_interval_episodes: int = 1  # It (train every It episodes)
    min_buffer_episodes: int = 5
    train_batches_per_interval: int = 50

    temperature: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stack_frames(frames: Deque[np.ndarray], history_len: int) -> jnp.ndarray:
    H, W, C = frames[-1].shape
    out = np.zeros((history_len, H, W, C), dtype=np.float32)
    start = history_len - len(frames)
    for i, f in enumerate(frames):
        out[start + i] = f
    out = out.reshape(history_len * C, H, W)
    return jnp.clip(jnp.array(out), 0.0, 1.0)


def sample_action(policy: np.ndarray, temperature: float) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    p = policy ** (1.0 / temperature)
    p = p / (p.sum() + 1e-8)
    return int(np.random.choice(len(policy), p=p))


def play_episode(
    env: MinatarEnv,
    net: MuZeroNet,
    params,
    mcts: MuZeroMCTS,
    history_len: int,
    max_steps: int,
    temperature: float,
) -> Episode:
    ep = Episode()
    obs = env.reset()

    frames: Deque[jnp.ndarray] = deque(maxlen=history_len)
    frames.append(obs)

    for _ in range(max_steps):
        stacked = stack_frames(frames, history_len)  # (C*hist, H, W)
        obs_t = jnp.expand_dims(stacked, axis=0)  # (1, C*hist, H, W)

        # Get hidden state from representation network
        hidden = net.apply(params, obs_t, method=lambda net, x: net.repr(x))
        hidden = hidden[0]  # Remove batch dimension

        policy, root_value = mcts.run(hidden)
        action = sample_action(np.array(policy), temperature)

        next_obs, reward, done = env.step(action)

        ep.add(
            StepData(
                obs=np.array(obs),  # Store as numpy for buffer
                action=action,
                reward=reward,
                policy=np.array(policy, dtype=np.float32),
                root_value=float(root_value),
                done=done,
            )
        )

        obs = next_obs
        frames.append(next_obs)
        if done:
            break

    return ep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="breakout")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    cfg = RunConfig(env_name=args.env, num_episodes=args.episodes)
    set_seed(cfg.seed)

    env = MinatarEnv(cfg.env_name, seed=cfg.seed)
    num_actions = env.num_actions()
    H, W, C = env.obs_shape

    train_cfg = TrainConfig(history_len=8, unroll_steps=5, td_steps=10, batch_size=32)
    
    # Initialize network
    net = MuZeroNet(obs_channels=C * train_cfg.history_len, num_actions=num_actions, hidden_dim=128)
    
    rng = jax.random.PRNGKey(cfg.seed)
    dummy_obs = jnp.zeros((1, C * train_cfg.history_len, H, W), dtype=jnp.float32)
    dummy_action = jnp.array([0], dtype=jnp.int32)
    params = net.init(rng, dummy_obs, dummy_action, method=MuZeroNet.init_all)

    # Create bound network for MCTS
    class BoundNetwork:
        def __init__(self, network, params):
            self.network = network
            self.params = params
        
        def pred(self, hidden):
            # Add batch dimension if needed
            if hidden.ndim == 1:
                hidden = jnp.expand_dims(hidden, axis=0)
            logits, value = self.network.apply(self.params, hidden, method=lambda net, h: net.pred(h))
            # Remove batch dimension
            return logits[0], value[0]
        
        def recurrent_inference(self, hidden, action):
            # Add batch dimensions if needed
            if hidden.ndim == 1:
                hidden = jnp.expand_dims(hidden, axis=0)
            if action.ndim == 0:
                action = jnp.expand_dims(action, axis=0)
            
            next_hidden, reward, next_logits, next_value = self.network.apply(
                self.params, hidden, action, 
                method=MuZeroNet.recurrent_inference
            )
            # Remove batch dimensions
            return next_hidden[0], reward[0], next_logits[0], next_value[0]
    
    bound_network = BoundNetwork(net, params)

    mcts_cfg = MCTSconfig(num_simulations=50, max_depth=5, discount=train_cfg.discount)
    mcts = MuZeroMCTS(mcts_cfg, num_actions=num_actions, network=bound_network)

    buffer = EpisodeBuffer(capacity=200)
    trainer = MuZeroTrainer(net, params, num_actions=num_actions, obs_shape_hwc=(H, W, C), cfg=train_cfg)

    for ep_i in range(cfg.num_episodes):
        episode = play_episode(
            env=env,
            net=net,
            params=trainer.state.params,  # Use current parameters
            mcts=mcts,
            history_len=train_cfg.history_len,
            max_steps=cfg.max_steps_per_episode,
            temperature=cfg.temperature,
        )

        ep_return = sum(s.reward for s in episode.steps)
        buffer.add_episode(episode)

        print(f"Episode {ep_i:04d} | steps={len(episode)} | return={ep_return:.2f} | buffer={len(buffer.episodes)}")

        if (ep_i + 1) % cfg.train_interval_episodes == 0 and len(buffer.episodes) >= cfg.min_buffer_episodes:
            stats = {}
            for _ in range(cfg.train_batches_per_interval):
                stats = trainer.train_batch(buffer)
            
            # Update bound network with new parameters
            bound_network.params = trainer.state.params
            
            print(
                "  train:",
                f"loss={stats['loss']:.4f}",
                f"policy={stats['policy_loss']:.4f}",
                f"value={stats['value_loss']:.4f}",
                f"reward={stats['reward_loss']:.4f}",
            )


if __name__ == "__main__":
    main()
