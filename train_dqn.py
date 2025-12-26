"""
Complete DQN Training Loop with Progress Tracking and Visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import minatar
import numpy as np
from datetime import datetime
import json
import pickle
from collections import defaultdict

from src.nn.network import DQN, DuelingDQN, DQNAgent, create_train_step_fn
from src.training.replay_buffer import ReplayBuffer


class DQNTrainer:
    """DQN Trainer with progress tracking and visualization."""
    
    def __init__(
        self,
        env_name: str = 'breakout',
        use_dueling: bool = False,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        batch_size: int = 32,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        train_freq: int = 4,
        learning_starts: int = 10000,
        seed: int = 42
    ):
        """Initialize the DQN trainer.
        
        Args:
            env_name: MinAtar environment name
            use_dueling: Whether to use Dueling DQN architecture
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay_steps: Number of steps to decay epsilon
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Frequency of target network updates
            train_freq: Frequency of training updates
            learning_starts: Number of steps before training starts
            seed: Random seed
        """
        self.env_name = env_name
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.seed = seed
        
        # Create environment
        self.env = minatar.Environment(env_name)
        self.env.seed(seed)
        
        obs_shape = self.env.state().shape
        action_dim = self.env.num_actions()
        
        # Create networks
        NetworkClass = DuelingDQN if use_dueling else DQN
        self.network = NetworkClass(action_dim=action_dim, features=(128, 128))
        
        # Create online and target agents
        self.online_agent = DQNAgent(
            network=self.network,
            observation_shape=obs_shape,
            action_dim=action_dim,
            learning_rate=learning_rate,
            seed=seed
        )
        
        # Target network (copy of online network)
        self.target_params = self.online_agent.params
        
        # Create training function
        self.train_step = create_train_step_fn(self.network, self.online_agent.optimizer)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
    
    def get_epsilon(self):
        """Get current epsilon value based on decay schedule."""
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        decay_ratio = self.total_steps / self.epsilon_decay_steps
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_ratio
    
    def compute_targets(self, rewards, next_states, dones):
        """Compute TD targets using target network.
        
        Args:
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        
        Returns:
            Target Q-values
        """
        # Get max Q-values from target network for next states
        next_q_values = self.network.apply(self.target_params, next_states)
        max_next_q = jnp.max(next_q_values, axis=1)
        
        # Compute targets: r + gamma * max_a' Q(s', a') * (1 - done)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        return targets
    
    def train_one_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute targets
        targets = self.compute_targets(rewards, next_states, dones)
        
        # Perform training step
        new_params, new_opt_state, loss = self.train_step(
            self.online_agent.params,
            self.online_agent.opt_state,
            states,
            actions,
            targets
        )
        
        # Update agent
        self.online_agent.params = new_params
        self.online_agent.opt_state = new_opt_state
        
        return float(loss)
    
    def run_episode(self, max_steps: int = 100000):
        """Run one episode and collect experience.
        
        Args:
            max_steps: Maximum steps per episode
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        self.env.reset()
        state = self.env.state()
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action
            epsilon = self.get_epsilon()
            action = self.online_agent.select_action(state, epsilon=epsilon)
            
            # Take action
            reward, done = self.env.act(action)
            next_state = self.env.state()
            
            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Training
            if self.total_steps >= self.learning_starts and self.total_steps % self.train_freq == 0:
                loss = self.train_one_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update target network
            if self.total_steps % self.target_update_freq == 0:
                self.target_params = self.online_agent.params
            
            state = next_state
            
            if done:
                break
        
        # Record metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        return episode_reward, episode_length, avg_loss, epsilon
    
    def train(self, num_episodes: int, eval_freq: int = 100, verbose: bool = True):
        """Train the DQN agent.
        
        Args:
            num_episodes: Number of episodes to train
            eval_freq: Frequency of evaluation and logging
            verbose: Whether to print progress
        
        Returns:
            Dictionary of training metrics
        """
        if verbose:
            print(f"Starting training for {num_episodes} episodes...")
            print(f"Environment: {self.env_name}")
            print(f"Network: {'Dueling DQN' if isinstance(self.network, DuelingDQN) else 'DQN'}")
            print(f"Total steps so far: {self.total_steps}")
            print("-" * 70)
        
        for episode in range(num_episodes):
            # Run episode
            reward, length, loss, epsilon = self.run_episode()
            
            # Log metrics
            self.metrics['episode'].append(self.episode_count)
            self.metrics['total_steps'].append(self.total_steps)
            self.metrics['reward'].append(reward)
            self.metrics['episode_length'].append(length)
            self.metrics['loss'].append(loss)
            self.metrics['epsilon'].append(epsilon)
            
            # Print progress
            if verbose and (episode + 1) % eval_freq == 0:
                recent_rewards = self.episode_rewards[-eval_freq:]
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                
                recent_losses = [l for l in self.metrics['loss'][-eval_freq:] if l > 0]
                avg_loss = np.mean(recent_losses) if recent_losses else 0.0
                
                print(f"Episode {self.episode_count:5d} | "
                      f"Steps: {self.total_steps:7d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Max Reward: {max_reward:3.0f} | "
                      f"Avg Loss: {avg_loss:6.4f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer):6d}")
        
        if verbose:
            print("-" * 70)
            print("Training completed!")
        
        return self.metrics
    
    def evaluate(self, num_episodes: int = 10, render: bool = False):
        """Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render (not supported by MinAtar)
        
        Returns:
            Dictionary with evaluation results
        """
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            self.env.reset()
            state = self.env.state()
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Greedy action selection (no exploration)
                action = self.online_agent.select_action(state, epsilon=0.0)
                reward, done = self.env.act(action)
                state = self.env.state()
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths),
        }
        
        return results
    
    def save(self, save_dir: str = 'checkpoints'):
        """Save the trained model and metrics.
        
        Args:
            save_dir: Directory to save checkpoint
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model parameters
        model_path = save_path / f"dqn_{self.env_name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'params': self.online_agent.params,
                'opt_state': self.online_agent.opt_state,
                'total_steps': self.total_steps,
                'episode_count': self.episode_count
            }, f)
        
        # Save metrics
        metrics_path = save_path / f"metrics_{self.env_name}_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            # Convert to serializable format
            metrics_dict = {k: [float(v) for v in vals] for k, vals in self.metrics.items()}
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        return model_path, metrics_path


def plot_training_progress(metrics: dict, save_path: str = None):
    """Plot training progress with multiple subplots.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    episodes = metrics['episode']
    rewards = metrics['reward']
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    window = min(100, len(rewards) // 10)
    if window > 0:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax = axes[0, 1]
    losses = [l for l in metrics['loss'] if l > 0]
    loss_episodes = [e for e, l in zip(episodes, metrics['loss']) if l > 0]
    
    if losses:
        ax.plot(loss_episodes, losses, alpha=0.3, color='orange')
        
        # Moving average
        window = min(100, len(losses) // 10)
        if window > 0:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(loss_episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    ax = axes[1, 0]
    steps = metrics['total_steps']
    epsilons = metrics['epsilon']
    ax.plot(steps, epsilons, color='green', linewidth=2)
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Episode Length
    ax = axes[1, 1]
    lengths = metrics['episode_length']
    ax.plot(episodes, lengths, alpha=0.3, color='purple')
    
    # Moving average
    window = min(100, len(lengths) // 10)
    if window > 0:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Training configuration
    config = {
        'env_name': 'breakout',
        'use_dueling': True,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 50000,
        'batch_size': 32,
        'buffer_size': 100000,
        'target_update_freq': 1000,
        'train_freq': 4,
        'learning_starts': 5000,
        'seed': 42
    }
    
    # Create trainer
    trainer = DQNTrainer(**config)
    
    # Train
    num_episodes = 1000
    metrics = trainer.train(num_episodes=num_episodes, eval_freq=50, verbose=True)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    eval_results = trainer.evaluate(num_episodes=10)
    print(f"Evaluation Results (10 episodes):")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Min/Max Reward: {eval_results['min_reward']:.0f} / {eval_results['max_reward']:.0f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
    
    # Save
    trainer.save()
    
    # Plot results
    plot_training_progress(metrics, save_path='training_progress.png')
