"""
Benchmark Evaluation Script for DQN and MuZero Agents
Runs multiple episodes and computes statistics without visualization.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import csv
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import minatar
import jax
import jax.numpy as jnp
from src.nn.network import DQN, DuelingDQN, DQNAgent
from src.muzero.network import RepresentationNet, DynamicsNet, PredictionNet
from src.muzero.mcts import MuZeroMCTS, MCTSconfig


def load_dqn_agent(checkpoint_path: str, env_name: str = "breakout", network_type: str = "dueling"):
    """Load a trained DQN agent from checkpoint."""
    import jax._src.core as jax_core
    original_init = jax_core.ShapedArray.__init__
    
    def patched_init(self, shape, dtype, weak_type=False, **kwargs):
        kwargs.pop('named_shape', None)
        original_init(self, shape, dtype, weak_type, **kwargs)
    
    jax_core.ShapedArray.__init__ = patched_init
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        jax_core.ShapedArray.__init__ = original_init
        raise ValueError(
            f"Failed to load checkpoint: {checkpoint_path}\n"
            f"Error: {e}\n"
            f"The checkpoint file may be corrupted or incomplete."
        )
    finally:
        jax_core.ShapedArray.__init__ = original_init
    
    env = minatar.Environment(env_name)
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    NetworkClass = DuelingDQN if network_type == 'dueling' else DQN
    network = NetworkClass(action_dim=action_dim, features=(128, 128))
    
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-4,
        seed=42
    )
    
    agent.params = checkpoint['params']
    
    return agent


def load_muzero_agent(checkpoint_path: str, env_name: str = "breakout"):
    """Load a trained MuZero agent from checkpoint."""
    import jax._src.core as jax_core
    original_init = jax_core.ShapedArray.__init__
    
    def patched_init(self, shape, dtype, weak_type=False, **kwargs):
        kwargs.pop('named_shape', None)
        original_init(self, shape, dtype, weak_type, **kwargs)
    
    jax_core.ShapedArray.__init__ = patched_init
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        jax_core.ShapedArray.__init__ = original_init
        raise ValueError(
            f"Failed to load checkpoint: {checkpoint_path}\n"
            f"Error: {e}\n"
            f"The checkpoint file may be corrupted or incomplete."
        )
    finally:
        jax_core.ShapedArray.__init__ = original_init
    
    if 'params' in checkpoint and isinstance(checkpoint['params'], dict):
        params = checkpoint['params']
        repr_params = params.get('repr', params.get('representation_params'))
        dyn_params = params.get('dyn', params.get('dynamics_params'))
        pred_params = params.get('pred', params.get('prediction_params'))
        config = checkpoint.get('config', {})
    else:
        repr_params = checkpoint.get('representation_params')
        dyn_params = checkpoint.get('dynamics_params')
        pred_params = checkpoint.get('prediction_params')
        config = checkpoint.get('config', {})
    
    env = minatar.Environment(env_name)
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    hidden_dim = config.get('hidden_dim', 128)
    
    if repr_params and 'Conv_0' in repr_params and 'kernel' in repr_params['Conv_0']:
        obs_channels = repr_params['Conv_0']['kernel'].shape[2]
        single_obs_channels = obs_shape[2]
        history_len = obs_channels // single_obs_channels
    else:
        obs_channels = obs_shape[2]
        history_len = 1
    
    repr_net = RepresentationNet(obs_channels=obs_channels, hidden_dim=hidden_dim)
    dyn_net = DynamicsNet(num_actions=action_dim, hidden_dim=hidden_dim)
    pred_net = PredictionNet(num_actions=action_dim, hidden_dim=hidden_dim)
    
    return repr_params, dyn_params, pred_params, (repr_net, dyn_net, pred_net), config, history_len


def evaluate_dqn(agent: DQNAgent, env_name: str, num_episodes: int = 50, 
                 epsilon: float = 0.0, seed: int = 42):
    """Evaluate DQN agent over multiple episodes."""
    print(f"\nEvaluating DQN Agent...")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Policy: {'Greedy' if epsilon == 0.0 else f'ε-greedy (ε={epsilon})'}")
    print("-" * 60)
    
    env = minatar.Environment(env_name)
    
    episode_rewards = []
    episode_steps = []
    episode_times = []
    
    start_total = time.time()
    
    for episode in range(num_episodes):
        start_episode = time.time()
        env.reset()
        state = env.state()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            
            reward, done = env.act(action)
            state = env.state()
            
            total_reward += reward
            steps += 1
        
        episode_time = time.time() - start_episode
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_times.append(episode_time)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
    
    total_time = time.time() - start_total
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'times': episode_times,
        'total_time': total_time
    }


def evaluate_muzero(repr_params, dyn_params, pred_params, networks, config, history_len,
                    env_name: str, num_episodes: int = 50, num_simulations: int = 50, seed: int = 42):
    """Evaluate MuZero agent over multiple episodes."""
    print(f"\nEvaluating MuZero Agent...")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"MCTS Simulations: {num_simulations}")
    print(f"Frame stacking: {history_len} frames")
    print("-" * 60)
    
    repr_net, dyn_net, pred_net = networks
    env = minatar.Environment(env_name)
    
    # Create wrapper
    class MuZeroNetWrapper:
        def __init__(self, repr_net, dyn_net, pred_net, repr_params, dyn_params, pred_params):
            self.repr_net = repr_net
            self.dyn_net = dyn_net
            self.pred_net = pred_net
            self.repr_params = repr_params
            self.dyn_params = dyn_params
            self.pred_params = pred_params
        
        def initial_inference(self, obs):
            hidden = self.repr_net.apply({'params': self.repr_params}, obs)
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, hidden)
            return hidden, policy_logits, value
        
        def recurrent_inference(self, hidden, action):
            next_hidden, reward = self.dyn_net.apply({'params': self.dyn_params}, hidden[None, ...], action[None, ...])
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, next_hidden)
            return next_hidden[0], reward[0], policy_logits[0], value[0]
        
        def pred(self, hidden):
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, hidden[None, ...])
            return policy_logits[0], value[0]
    
    muzero_net = MuZeroNetWrapper(repr_net, dyn_net, pred_net, repr_params, dyn_params, pred_params)
    
    mcts_config = MCTSconfig(
        num_simulations=num_simulations,
        pb_c_base=config.get('pb_c_base', 19652),
        pb_c_init=config.get('c_puct', 1.25),
        dirichlet_alpha=config.get('dirichlet_alpha', 0.25),
        root_exploration_frac=0.0
    )
    
    mcts = MuZeroMCTS(config=mcts_config, num_actions=env.num_actions(), network=muzero_net)
    
    episode_rewards = []
    episode_steps = []
    episode_times = []
    
    start_total = time.time()
    
    for episode in range(num_episodes):
        start_episode = time.time()
        env.reset()
        obs = env.state()
        
        from collections import deque
        frame_history = deque(maxlen=history_len)
        for _ in range(history_len):
            frame_history.append(obs)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            stacked_obs = jnp.concatenate(list(frame_history), axis=-1)
            obs_jax = jnp.array(stacked_obs, dtype=jnp.float32)
            obs_batch = jnp.transpose(obs_jax, (2, 0, 1))[None, ...]
            
            hidden = repr_net.apply({'params': repr_params}, obs_batch)
            policy, _ = mcts.run(hidden[0])
            action = int(jnp.argmax(policy))
            
            reward, done = env.act(action)
            obs = env.state()
            frame_history.append(obs)
            
            total_reward += reward
            steps += 1
        
        episode_time = time.time() - start_episode
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_times.append(episode_time)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
    
    total_time = time.time() - start_total
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'times': episode_times,
        'total_time': total_time
    }


def print_statistics(results, agent_type):
    """Print evaluation statistics."""
    rewards = results['rewards']
    steps = results['steps']
    times = results['times']
    total_time = results['total_time']
    
    print(f"\n{'='*60}")
    print(f"{agent_type} Evaluation Results")
    print(f"{'='*60}")
    print(f"\nReward Statistics:")
    print(f"  Average:  {np.mean(rewards):.2f}")
    print(f"  Best:     {np.max(rewards):.2f}")
    print(f"  Worst:    {np.min(rewards):.2f}")
    print(f"  Std Dev:  {np.std(rewards):.2f}")
    
    print(f"\nSteps Statistics:")
    print(f"  Average:  {np.mean(steps):.2f}")
    print(f"  Best:     {np.min(steps):.0f}")
    print(f"  Worst:    {np.max(steps):.0f}")
    
    print(f"\nTime Statistics:")
    print(f"  Avg per episode:  {np.mean(times):.3f}s")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"{'='*60}\n")


def save_results_csv(results, agent_type, env_name, checkpoint_path, output_path):
    """Save evaluation results to CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    rewards = results['rewards']
    steps = results['steps']
    
    # Summary statistics
    summary = {
        'timestamp': timestamp,
        'agent_type': agent_type,
        'environment': env_name,
        'checkpoint': checkpoint_path,
        'num_episodes': len(rewards),
        'avg_reward': np.mean(rewards),
        'best_reward': np.max(rewards),
        'worst_reward': np.min(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps),
        'min_steps': np.min(steps),
        'max_steps': np.max(steps),
        'avg_time_per_episode': np.mean(results['times']),
        'total_time': results['total_time']
    }
    
    # Save summary
    summary_path = Path(output_path) / f"{agent_type}_{env_name}_{timestamp}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    
    # Save detailed results
    detail_path = Path(output_path) / f"{agent_type}_{env_name}_{timestamp}_details.csv"
    
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Steps', 'Time'])
        for i, (r, s, t) in enumerate(zip(rewards, steps, results['times'])):
            writer.writerow([i + 1, r, s, t])
    
    print(f"Results saved:")
    print(f"  Summary: {summary_path}")
    print(f"  Details: {detail_path}")
    
    return summary_path, detail_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for DQN or MuZero agents"
    )
    
    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['dqn', 'muzero'],
        required=True,
        help='Type of agent to evaluate'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pkl file)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='breakout',
        choices=['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders'],
        help='MinAtar environment name (default: breakout)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of episodes to evaluate (default: 50)'
    )
    
    parser.add_argument(
        '--network-type',
        type=str,
        default='dueling',
        choices=['standard', 'dueling'],
        help='DQN network type (default: dueling, only for DQN agent)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0,
        help='Exploration rate for DQN (default: 0.0 = greedy, only for DQN agent)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='MCTS simulations per action (default: 50, only for MuZero agent)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Directory to save CSV results (default: benchmark_results)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MinAtar Agent Benchmark Evaluation")
    print("="*60)
    
    if args.agent_type == 'dqn':
        agent = load_dqn_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            network_type=args.network_type
        )
        
        results = evaluate_dqn(
            agent=agent,
            env_name=args.env,
            num_episodes=args.episodes,
            epsilon=args.epsilon,
            seed=args.seed
        )
        
        print_statistics(results, "DQN")
        save_results_csv(results, "dqn", args.env, args.checkpoint, args.output_dir)
    
    elif args.agent_type == 'muzero':
        repr_params, dyn_params, pred_params, networks, config, history_len = load_muzero_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env
        )
        
        results = evaluate_muzero(
            repr_params=repr_params,
            dyn_params=dyn_params,
            pred_params=pred_params,
            networks=networks,
            config=config,
            history_len=history_len,
            env_name=args.env,
            num_episodes=args.episodes,
            num_simulations=args.simulations,
            seed=args.seed
        )
        
        print_statistics(results, "MuZero")
        save_results_csv(results, "muzero", args.env, args.checkpoint, args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
