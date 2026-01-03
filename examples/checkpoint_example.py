"""
Example: Training with Checkpoints and Resume Capability
Demonstrates how to use the new checkpoint and resume features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_dqn import DQNTrainer, plot_training_progress


def example_basic_training_with_best_model():
    """Example 1: Basic training with best model tracking."""
    
    print("=" * 70)
    print("EXAMPLE 1: Training with Best Model Tracking")
    print("=" * 70)
    
    # Create trainer
    trainer = DQNTrainer(
        env_name='breakout',
        use_dueling=True,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=10000,
        learning_starts=1000,
        seed=42
    )
    
    # Train with best model tracking (short demo)
    metrics = trainer.train(
        num_episodes=100,
        eval_freq=20,
        verbose=True,
        save_best=True,
        best_metric='avg_reward',
        checkpoint_dir='checkpoints/demo'
    )
    
    # Save final checkpoint
    model_path, _ = trainer.save(save_dir='checkpoints/demo', prefix='demo_final')
    
    print(f"\n✓ Training complete!")
    print(f"✓ Final checkpoint: {model_path}")
    print(f"✓ Best average reward: {trainer.best_avg_reward:.2f}")
    
    return model_path


def example_resume_training(checkpoint_path):
    """Example 2: Resume training from a checkpoint."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Resuming Training from Checkpoint")
    print("=" * 70)
    
    # Create new trainer
    trainer = DQNTrainer(
        env_name='breakout',
        use_dueling=True,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=10000,
        learning_starts=1000,
        seed=42
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer.load(checkpoint_path)
    
    print(f"Resumed from episode {trainer.episode_count}, step {trainer.total_steps}")
    print(f"Current epsilon: {trainer.get_epsilon():.4f}")
    print(f"Buffer size: {len(trainer.replay_buffer)}")
    print(f"Previous best avg reward: {trainer.best_avg_reward:.2f}")
    
    # Continue training
    print("\nContinuing training for 50 more episodes...")
    metrics = trainer.train(
        num_episodes=50,
        eval_freq=10,
        verbose=True,
        save_best=True,
        best_metric='avg_reward',
        checkpoint_dir='checkpoints/demo'
    )
    
    # Save final checkpoint
    final_path, _ = trainer.save(save_dir='checkpoints/demo', prefix='demo_resumed')
    
    print(f"\n✓ Resumed training complete!")
    print(f"✓ Total episodes: {trainer.episode_count}")
    print(f"✓ Total steps: {trainer.total_steps}")
    print(f"✓ Best average reward: {trainer.best_avg_reward:.2f}")
    print(f"✓ Final checkpoint: {final_path}")


def example_different_metrics():
    """Example 3: Compare different best model metrics."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Different Best Model Metrics")
    print("=" * 70)
    
    metrics_to_test = ['avg_reward', 'max_reward', 'total_reward']
    
    for metric in metrics_to_test:
        print(f"\n--- Training with best_metric='{metric}' ---")
        
        trainer = DQNTrainer(
            env_name='breakout',
            use_dueling=True,
            batch_size=32,
            buffer_size=5000,
            learning_starts=500,
            seed=42
        )
        
        trainer.train(
            num_episodes=30,
            eval_freq=10,
            verbose=True,
            save_best=True,
            best_metric=metric,
            checkpoint_dir=f'checkpoints/demo/{metric}'
        )
        
        if metric == 'avg_reward':
            print(f"Best average reward: {trainer.best_avg_reward:.2f}")
        elif metric == 'max_reward':
            print(f"Best max reward: {trainer.best_max_reward:.2f}")
        elif metric == 'total_reward':
            print(f"Best total reward: {trainer.best_total_reward:.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DQN CHECKPOINT AND RESUME EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates:")
    print("  1. Training with automatic best model saving")
    print("  2. Resuming training from a checkpoint")
    print("  3. Using different metrics for best model selection")
    print()
    
    # Example 1: Train with best model tracking
    checkpoint_path = example_basic_training_with_best_model()
    
    # Example 2: Resume from checkpoint
    example_resume_training(checkpoint_path)
    
    # Example 3: Different metrics (optional, commented out for speed)
    # example_different_metrics()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nCheckpoints saved in: checkpoints/demo/")
    print("Try resuming with: python train_with_config.py --resume <checkpoint_path>")
    print()
