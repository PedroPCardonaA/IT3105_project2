#!/usr/bin/env python3
"""
Configuration Management Tool
Utility for managing DQN training configurations
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    load_config,
    print_config,
    list_available_configs,
    create_custom_config
)


def list_configs():
    """List all available configuration files."""
    print("\n" + "=" * 70)
    print("AVAILABLE CONFIGURATIONS")
    print("=" * 70 + "\n")
    
    configs = list_available_configs()
    
    if not configs:
        print("  No configurations found in configs/ directory")
        return
    
    for config_name in configs:
        config_path = f"configs/{config_name}.yaml"
        print(f"📄 {config_name}")
        print(f"   Path: {config_path}")
        
        # Try to load and show brief info
        try:
            config = load_config(config_path)
            env = config['environment']['name']
            episodes = config['training']['num_episodes']
            net_type = config['network']['type']
            print(f"   → {env} | {episodes} episodes | {net_type} network")
        except Exception as e:
            print(f"   → Error loading: {e}")
        print()
    
    print("Usage: python train_with_config.py --config configs/<name>.yaml\n")


def show_config(config_path):
    """Display a configuration file."""
    print(f"\nLoading: {config_path}\n")
    
    try:
        config = load_config(config_path)
        print_config(config)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("   Use --list to see available configs")
    except Exception as e:
        print(f"❌ Error loading config: {e}")


def create_config(output_path, env_name, num_episodes, learning_rate, network_type):
    """Create a new configuration file."""
    print(f"\nCreating new config: {output_path}")
    
    kwargs = {}
    if env_name:
        kwargs['env_name'] = env_name
    if num_episodes:
        kwargs['num_episodes'] = num_episodes
    if learning_rate:
        kwargs['learning_rate'] = learning_rate
    if network_type:
        kwargs['network_type'] = network_type
    
    try:
        create_custom_config(output_path, **kwargs)
        print("✓ Config created successfully!")
        print(f"\nTo use: python train_with_config.py --config {output_path}")
    except Exception as e:
        print(f"❌ Error creating config: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='DQN Configuration Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all configs
  python config_manager.py --list
  
  # Show a config
  python config_manager.py --show configs/default.yaml
  
  # Create custom config
  python config_manager.py --create configs/my_config.yaml \\
      --env asterix --episodes 2000 --lr 0.0005 --network dueling
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available configurations'
    )
    
    parser.add_argument(
        '--show', '-s',
        type=str,
        metavar='CONFIG',
        help='Show configuration details'
    )
    
    parser.add_argument(
        '--create', '-c',
        type=str,
        metavar='OUTPUT',
        help='Create a new configuration file'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        choices=['breakout', 'asterix', 'freeway', 'seaquest', 'space_invaders'],
        help='Environment name (for --create)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        help='Number of episodes (for --create)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (for --create)'
    )
    
    parser.add_argument(
        '--network',
        type=str,
        choices=['standard', 'dueling'],
        help='Network type (for --create)'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.list:
        list_configs()
    elif args.show:
        show_config(args.show)
    elif args.create:
        create_config(
            args.create,
            args.env,
            args.episodes,
            args.lr,
            args.network
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
