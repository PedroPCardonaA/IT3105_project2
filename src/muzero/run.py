from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Any
import argparse
import random
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from pathlib import Path

from minatar_env import MinatarEnv
from network import MuZeroNet
from mcts import MuZeroMCTS, MCTSconfig
from buffer import Episode, EpisodeBuffer, StepData
from trainer import MuZeroTrainer, TrainConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print(f"Available configs in configs/:")
        import os
        if os.path.exists('configs'):
            for f in os.listdir('configs'):
                if f.startswith('muzero') and f.endswith('.yaml'):
                    print(f"  - configs/{f}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_path}': {e}")
        raise


@dataclass(frozen=True)
class RunConfig:
    env_name: str
    seed: int
    num_episodes: int
    max_steps_per_episode: int
    train_interval_episodes: int
    min_buffer_episodes: int
    train_batches_per_interval: int
    temperature: float
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RunConfig':
        """Create RunConfig from configuration dictionary."""
        return cls(
            env_name=config['environment']['name'],
            seed=config['environment']['seed'],
            num_episodes=config['training']['num_episodes'],
            max_steps_per_episode=config['environment']['max_steps_per_episode'],
            train_interval_episodes=config['training']['train_interval_episodes'],
            min_buffer_episodes=config['training']['min_buffer_episodes'],
            train_batches_per_interval=config['training']['train_batches_per_interval'],
            temperature=config['action_selection']['temperature'],
        )


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
    parser = argparse.ArgumentParser(description='Train MuZero on MinAtar environments')
    parser.add_argument("--config", type=str, default="configs/muzero_default.yaml",
                        help="Path to config file")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name (overrides config)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load configuration from YAML
    config_dict = load_config(args.config)
    
    # Override with command-line arguments if provided
    if args.env is not None:
        config_dict['environment']['name'] = args.env
    if args.episodes is not None:
        config_dict['training']['num_episodes'] = args.episodes
    if args.seed is not None:
        config_dict['environment']['seed'] = args.seed
    
    # Create run configuration
    cfg = RunConfig.from_dict(config_dict)
    set_seed(cfg.seed)

    # Initialize environment
    env = MinatarEnv(cfg.env_name, seed=cfg.seed)
    num_actions = env.num_actions()
    H, W, C = env.obs_shape

    # Create training configuration from YAML
    train_cfg = TrainConfig(
        history_len=config_dict['training']['history_len'],
        unroll_steps=config_dict['training']['unroll_steps'],
        td_steps=config_dict['training']['td_steps'],
        discount=config_dict['training']['discount'],
        batch_size=config_dict['training']['batch_size'],
        lr=config_dict['training']['learning_rate'],
        weight_decay=config_dict['training']['weight_decay'],
        grad_clip=config_dict['training']['grad_clip'],
        policy_loss_weight=config_dict['training']['policy_loss_weight'],
        value_loss_weight=config_dict['training']['value_loss_weight'],
        reward_loss_weight=config_dict['training']['reward_loss_weight'],
    )
    
    # Initialize network
    hidden_dim = config_dict['network']['hidden_dim']
    net = MuZeroNet(
        obs_channels=C * train_cfg.history_len, 
        num_actions=num_actions, 
        hidden_dim=hidden_dim
    )
    
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

    # Create MCTS configuration from YAML
    mcts_cfg = MCTSconfig(
        num_simulations=config_dict['mcts']['num_simulations'],
        max_depth=config_dict['mcts']['max_depth'],
        discount=train_cfg.discount,
        pb_c_base=config_dict['mcts']['pb_c_base'],
        pb_c_init=config_dict['mcts']['pb_c_init'],
        dirichlet_alpha=config_dict['mcts']['dirichlet_alpha'],
        root_exploration_frac=config_dict['mcts']['root_exploration_frac'],
    )
    mcts = MuZeroMCTS(mcts_cfg, num_actions=num_actions, network=bound_network)

    # Create buffer from YAML config
    buffer = EpisodeBuffer(capacity=config_dict['buffer']['capacity'])
    trainer = MuZeroTrainer(net, params, num_actions=num_actions, obs_shape_hwc=(H, W, C), cfg=train_cfg)

    # Print configuration summary
    if config_dict.get('logging', {}).get('verbose', True):
        print(f"\n{'='*60}")
        print(f"MuZero Training Configuration")
        print(f"{'='*60}")
        print(f"Environment: {cfg.env_name}")
        print(f"Seed: {cfg.seed}")
        print(f"Episodes: {cfg.num_episodes}")
        print(f"Network hidden dim: {hidden_dim}")
        print(f"MCTS simulations: {mcts_cfg.num_simulations}")
        print(f"Batch size: {train_cfg.batch_size}")
        print(f"Learning rate: {train_cfg.lr}")
        print(f"{'='*60}\n")

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
