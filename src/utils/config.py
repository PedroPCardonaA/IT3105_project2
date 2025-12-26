"""
Configuration utilities for loading YAML config files
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary with configuration parameters
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def config_to_trainer_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML config to DQNTrainer parameters.
    
    Args:
        config: Configuration dictionary from YAML
    
    Returns:
        Dictionary with parameters for DQNTrainer
    """
    # Extract network type
    use_dueling = config['network']['type'].lower() == 'dueling'
    
    # Build trainer parameters
    trainer_params = {
        'env_name': config['environment']['name'],
        'seed': config['environment']['seed'],
        'use_dueling': use_dueling,
        'learning_rate': config['training']['learning_rate'],
        'gamma': config['training']['gamma'],
        'batch_size': config['training']['batch_size'],
        'buffer_size': config['training']['buffer_size'],
        'learning_starts': config['training']['learning_starts'],
        'train_freq': config['training']['train_freq'],
        'target_update_freq': config['training']['target_update_freq'],
        'epsilon_start': config['exploration']['epsilon_start'],
        'epsilon_end': config['exploration']['epsilon_end'],
        'epsilon_decay_steps': config['exploration']['epsilon_decay_steps'],
    }
    
    return trainer_params


def print_config(config: Dict[str, Any]):
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    
    print("\n[Environment]")
    for key, value in config['environment'].items():
        print(f"  {key:25s}: {value}")
    
    print("\n[Network]")
    for key, value in config['network'].items():
        print(f"  {key:25s}: {value}")
    
    print("\n[Training]")
    for key, value in config['training'].items():
        print(f"  {key:25s}: {value}")
    
    print("\n[Exploration]")
    for key, value in config['exploration'].items():
        print(f"  {key:25s}: {value}")
    
    print("\n[Logging]")
    for key, value in config['logging'].items():
        print(f"  {key:25s}: {value}")
    
    print("=" * 70)


def list_available_configs(config_dir: str = 'configs') -> list:
    """List all available configuration files.
    
    Args:
        config_dir: Directory containing config files
    
    Returns:
        List of available config file names
    """
    config_path = Path(config_dir)
    
    if not config_path.exists():
        return []
    
    configs = sorted(config_path.glob('*.yaml'))
    return [c.stem for c in configs]


def create_custom_config(output_path: str, **kwargs):
    """Create a custom configuration file.
    
    Args:
        output_path: Path to save the new config file
        **kwargs: Configuration parameters to override
    
    Example:
        create_custom_config(
            'configs/my_config.yaml',
            env_name='asterix',
            learning_rate=0.0005,
            num_episodes=2000
        )
    """
    # Load default config as template
    default_config = load_config('configs/default.yaml')
    
    # Update with custom parameters
    # Map simple parameter names to nested structure
    param_map = {
        'env_name': ('environment', 'name'),
        'seed': ('environment', 'seed'),
        'network_type': ('network', 'type'),
        'features': ('network', 'features'),
        'num_episodes': ('training', 'num_episodes'),
        'learning_rate': ('training', 'learning_rate'),
        'gamma': ('training', 'gamma'),
        'batch_size': ('training', 'batch_size'),
        'buffer_size': ('training', 'buffer_size'),
        'learning_starts': ('training', 'learning_starts'),
        'train_freq': ('training', 'train_freq'),
        'target_update_freq': ('training', 'target_update_freq'),
        'epsilon_start': ('exploration', 'epsilon_start'),
        'epsilon_end': ('exploration', 'epsilon_end'),
        'epsilon_decay_steps': ('exploration', 'epsilon_decay_steps'),
        'eval_freq': ('logging', 'eval_freq'),
        'eval_episodes': ('logging', 'eval_episodes'),
    }
    
    for param, value in kwargs.items():
        if param in param_map:
            section, key = param_map[param]
            default_config[section][key] = value
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Custom config saved to: {output_path}")
