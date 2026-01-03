#!/usr/bin/env python3
"""
Quick Evaluation Script
Automatically finds and evaluates the most recent checkpoint
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_model import evaluate_model, print_results, save_evaluation_results


def find_latest_checkpoint(checkpoint_dir: str = 'checkpoints'):
    """Find the most recent checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = sorted(checkpoint_path.glob('best_avg_reward_**.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        return None
    
    return str(checkpoints[0])


def main():
    """Main function."""
    print("=" * 70)
    print("QUICK GREEDY EVALUATION")
    print("=" * 70)
    
    # Find latest checkpoint
    print("\nSearching for latest checkpoint...")
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("❌ No checkpoints found in checkpoints/ directory")
        print("\nTrain a model first:")
        print("  python train_with_config.py --config configs/demo.yaml")
        return
    
    print(f"✓ Found: {checkpoint_path}")
    
    # Run evaluation
    print("\nRunning greedy evaluation (100 episodes)...")
    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        num_episodes=100,
        env_name=None,  # Auto-detect
        network_type='dueling',
        verbose=True,
        seed=42
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = 'evaluation_results.json'
    save_evaluation_results(results, output_path)
    
    print("\n💡 TIP: For more control, use:")
    print(f"  python evaluate_model.py {checkpoint_path} --episodes 200")


if __name__ == "__main__":
    main()
