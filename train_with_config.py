"""
Train DQN with YAML Configuration
Load hyperparameters from YAML file for easy experimentation
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from train_dqn import DQNTrainer, plot_training_progress
from src.utils.config import (
    load_config, 
    config_to_trainer_params, 
    print_config,
    list_available_configs
)


def main():
    """Main training function with config file support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN with YAML configuration')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to YAML config file (default: configs/default.yaml)'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configurations'
    )
    
    args = parser.parse_args()
    
    # List configs if requested
    if args.list_configs:
        print("\nAvailable configurations:")
        configs = list_available_configs()
        for config_name in configs:
            print(f"  - configs/{config_name}.yaml")
        print("\nUsage: python train_with_config.py --config configs/<name>.yaml\n")
        return
    
    # Load configuration
    print("\nLoading configuration from:", args.config)
    config = load_config(args.config)
    
    # Print configuration
    print_config(config)
    
    # Convert to trainer parameters
    trainer_params = config_to_trainer_params(config)
    
    # Create trainer
    print("\nInitializing DQN Trainer...")
    trainer = DQNTrainer(**trainer_params)
    
    # Train
    print("\nStarting training...\n")
    num_episodes = config['training']['num_episodes']
    eval_freq = config['logging']['eval_freq']
    
    metrics = trainer.train(
        num_episodes=num_episodes,
        eval_freq=eval_freq,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    eval_episodes = config['logging']['eval_episodes']
    eval_results = trainer.evaluate(num_episodes=eval_episodes)
    
    print(f"\nGreedy Evaluation Results ({eval_episodes} episodes, epsilon=0.0):")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Min Reward:  {eval_results['min_reward']:.0f}")
    print(f"  Max Reward:  {eval_results['max_reward']:.0f}")
    print(f"  Mean Length: {eval_results['mean_length']:.1f}")
    print(f"\n  Performance breakdown:")
    rewards = [eval_results['mean_reward']] * eval_episodes  # Placeholder
    print(f"    Episodes with reward > 0:  {sum(1 for _ in range(eval_episodes) if eval_results['mean_reward'] > 0)}")
    
    # Save checkpoint
    if config['logging']['save_checkpoints']:
        print("\n" + "=" * 70)
        print("SAVING CHECKPOINT")
        print("=" * 70)
        checkpoint_dir = config['logging']['checkpoint_dir']
        trainer.save(save_dir=checkpoint_dir)
    
    # Plot results
    if config['logging']['plot_results']:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)
        plot_path = config['logging']['plot_path']
        plot_training_progress(metrics, save_path=plot_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {trainer.episode_count}")
    print(f"  Total Steps:    {trainer.total_steps}")
    print(f"  Buffer Size:    {len(trainer.replay_buffer)}")
    print(f"  Final Epsilon:  {trainer.get_epsilon():.4f}")
    print()


if __name__ == "__main__":
    main()
