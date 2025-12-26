"""
Evaluate a Trained DQN Model
Test agent performance with fully greedy policy (epsilon=0)
"""

import sys
from pathlib import Path
import argparse
import pickle
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import minatar
from src.nn.network import DQN, DuelingDQN, DQNAgent
from src.utils.config import load_config, config_to_trainer_params


def load_checkpoint(checkpoint_path: str):
    """Load a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pkl file
    
    Returns:
        Dictionary with checkpoint data
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def find_matching_config(checkpoint_path: str):
    """Try to find the matching metrics file to determine configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Environment name from checkpoint filename, or None
    """
    # Extract env name from checkpoint filename
    # Format: dqn_{env_name}_{timestamp}.pkl
    filename = Path(checkpoint_path).stem
    parts = filename.split('_')
    
    if len(parts) >= 2:
        return parts[1]  # env_name
    
    return None


def evaluate_model(
    checkpoint_path: str,
    num_episodes: int = 100,
    env_name: str = None,
    network_type: str = 'dueling',
    verbose: bool = True,
    seed: int = 42
):
    """Evaluate a trained model with greedy policy.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        env_name: Environment name (auto-detect if None)
        network_type: 'standard' or 'dueling'
        verbose: Whether to print progress
        seed: Random seed
    
    Returns:
        Dictionary with evaluation results
    """
    # Load checkpoint
    if verbose:
        print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Auto-detect environment if not specified
    if env_name is None:
        env_name = find_matching_config(checkpoint_path)
        if env_name is None:
            env_name = 'breakout'  # Default
        if verbose:
            print(f"Auto-detected environment: {env_name}")
    
    # Create environment
    env = minatar.Environment(env_name)
    env.seed(seed)
    env.reset()
    
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    if verbose:
        print(f"Environment: {env_name}")
        print(f"Observation shape: {obs_shape}")
        print(f"Action dimension: {action_dim}")
    
    # Create network
    NetworkClass = DuelingDQN if network_type == 'dueling' else DQN
    network = NetworkClass(action_dim=action_dim, features=(128, 128))
    
    # Create agent and load parameters
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-4,  # Not used for evaluation
        seed=seed
    )
    
    # Load trained parameters
    agent.params = checkpoint['params']
    
    if verbose:
        print(f"Network: {network_type.capitalize()} DQN")
        print(f"Total training steps: {checkpoint.get('total_steps', 'unknown')}")
        print(f"Episodes trained: {checkpoint.get('episode_count', 'unknown')}")
        print(f"\nEvaluating with {num_episodes} episodes (GREEDY POLICY, epsilon=0)...")
        print("=" * 70)
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_details = []
    
    for episode in range(num_episodes):
        env.reset()
        state = env.state()
        
        episode_reward = 0
        episode_length = 0
        actions_taken = []
        
        max_steps = 10000  # Safety limit
        
        for step in range(max_steps):
            # GREEDY ACTION SELECTION (epsilon=0)
            action = agent.select_action(state, epsilon=0.0)
            actions_taken.append(action)
            
            # Take action
            reward, done = env.act(action)
            state = env.state()
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_details.append({
            'reward': episode_reward,
            'length': episode_length,
            'actions': actions_taken
        })
        
        # Print progress every 10 episodes
        if verbose and (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            print(f"Episodes {episode-8:3d}-{episode+1:3d} | "
                  f"Mean Reward: {np.mean(recent_rewards):6.2f} | "
                  f"Mean Length: {np.mean(episode_lengths[-10:]):6.1f}")
    
    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_length': np.min(episode_lengths),
        'max_length': np.max(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_details': episode_details,
        'checkpoint_info': {
            'total_steps': checkpoint.get('total_steps'),
            'episode_count': checkpoint.get('episode_count')
        }
    }
    
    return results


def print_results(results: dict):
    """Print evaluation results in a nice format.
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("GREEDY EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nEpisodes Evaluated: {results['num_episodes']}")
    print(f"Policy: FULLY GREEDY (epsilon = 0.0)")
    
    print("\n📊 REWARD STATISTICS")
    print(f"  Mean   ± Std:  {results['mean_reward']:6.2f} ± {results['std_reward']:5.2f}")
    print(f"  Median:        {results['median_reward']:6.2f}")
    print(f"  Min - Max:     {results['min_reward']:6.0f} - {results['max_reward']:6.0f}")
    
    print("\n📏 EPISODE LENGTH STATISTICS")
    print(f"  Mean   ± Std:  {results['mean_length']:6.1f} ± {results['std_length']:5.1f}")
    print(f"  Min - Max:     {results['min_length']:6.0f} - {results['max_length']:6.0f}")
    
    print("\n🎯 PERFORMANCE BREAKDOWN")
    rewards = results['episode_rewards']
    print(f"  Episodes with reward = 0:   {sum(1 for r in rewards if r == 0):4d} ({sum(1 for r in rewards if r == 0)/len(rewards)*100:5.1f}%)")
    print(f"  Episodes with reward > 0:   {sum(1 for r in rewards if r > 0):4d} ({sum(1 for r in rewards if r > 0)/len(rewards)*100:5.1f}%)")
    print(f"  Episodes with reward ≥ 5:   {sum(1 for r in rewards if r >= 5):4d} ({sum(1 for r in rewards if r >= 5)/len(rewards)*100:5.1f}%)")
    print(f"  Episodes with reward ≥ 10:  {sum(1 for r in rewards if r >= 10):4d} ({sum(1 for r in rewards if r >= 10)/len(rewards)*100:5.1f}%)")
    
    print("\n📈 REWARD DISTRIBUTION")
    bins = [0, 1, 2, 5, 10, 20, 50, 100]
    for i in range(len(bins) - 1):
        count = sum(1 for r in rewards if bins[i] <= r < bins[i+1])
        print(f"  [{bins[i]:3d}, {bins[i+1]:3d}): {count:4d} episodes")
    count = sum(1 for r in rewards if r >= bins[-1])
    if count > 0:
        print(f"  [{bins[-1]:3d}, ∞  ): {count:4d} episodes")
    
    print("\n🏆 TOP 5 EPISODES")
    top_episodes = sorted(enumerate(rewards), key=lambda x: x[1], reverse=True)[:5]
    for rank, (ep_idx, reward) in enumerate(top_episodes, 1):
        length = results['episode_lengths'][ep_idx]
        print(f"  {rank}. Episode {ep_idx+1:4d}: Reward = {reward:6.1f}, Length = {length:4d}")
    
    if results['checkpoint_info']['total_steps']:
        print("\n📚 TRAINING INFO")
        print(f"  Total Training Steps:    {results['checkpoint_info']['total_steps']}")
        print(f"  Total Training Episodes: {results['checkpoint_info']['episode_count']}")
    
    print("\n" + "=" * 70)


def save_evaluation_results(results: dict, output_path: str):
    """Save evaluation results to JSON.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save JSON file
    """
    # Convert to serializable format
    output = {
        'num_episodes': results['num_episodes'],
        'statistics': {
            'mean_reward': float(results['mean_reward']),
            'std_reward': float(results['std_reward']),
            'min_reward': float(results['min_reward']),
            'max_reward': float(results['max_reward']),
            'median_reward': float(results['median_reward']),
            'mean_length': float(results['mean_length']),
            'std_length': float(results['std_length']),
            'min_length': float(results['min_length']),
            'max_length': float(results['max_length']),
        },
        'episode_rewards': [float(r) for r in results['episode_rewards']],
        'episode_lengths': [int(l) for l in results['episode_lengths']],
        'checkpoint_info': results['checkpoint_info']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN model with greedy policy')
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint (.pkl file)'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        choices=['breakout', 'asterix', 'freeway', 'seaquest', 'space_invaders'],
        help='Environment name (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--network',
        type=str,
        choices=['standard', 'dueling'],
        default='dueling',
        help='Network type (default: dueling)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (less output)'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('dqn_*.pkl'))
            for cp in checkpoints:
                print(f"  {cp}")
        return
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        env_name=args.env,
        network_type=args.network,
        verbose=not args.quiet,
        seed=args.seed
    )
    
    # Print results
    print_results(results)
    
    # Save if requested
    if args.save:
        save_evaluation_results(results, args.save)


if __name__ == "__main__":
    main()
