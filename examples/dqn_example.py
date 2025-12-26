"""
Example usage of JAX DQN network with MinAtar
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import minatar
from src.nn.network import DQN, DuelingDQN, DQNAgent, create_dqn_loss_fn, create_train_step_fn


def example_basic_usage():
    """Basic example of creating and using a DQN network."""
    print("=== Basic DQN Usage ===\n")
    
    # Create MinAtar environment
    env = minatar.Environment('breakout')
    env.reset()
    
    # Get observation shape and action dimension
    obs_shape = env.state().shape  # (10, 10, 4) for MinAtar
    action_dim = env.num_actions()
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}\n")
    
    # Create DQN network
    network = DQN(action_dim=action_dim, features=(128, 128))
    
    # Create agent
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-4,
        seed=42
    )
    
    # Get current observation
    obs = env.state()
    
    # Get Q-values
    q_values = agent.get_q_values(obs)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}\n")
    
    # Select action (greedy)
    action = agent.select_action(obs, epsilon=0.0)
    print(f"Greedy action: {action}")
    
    # Select action (epsilon-greedy with exploration)
    action = agent.select_action(obs, epsilon=0.1)
    print(f"Epsilon-greedy action (ε=0.1): {action}\n")


def example_dueling_dqn():
    """Example using Dueling DQN architecture."""
    print("=== Dueling DQN Usage ===\n")
    
    # Create MinAtar environment
    env = minatar.Environment('breakout')
    env.reset()
    
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    # Create Dueling DQN network
    network = DuelingDQN(action_dim=action_dim, features=(128, 128))
    
    # Create agent
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-4,
        seed=42
    )
    
    obs = env.state()
    q_values = agent.get_q_values(obs)
    print(f"Dueling DQN Q-values: {q_values}\n")


def example_training_step():
    """Example of a training step with batch of transitions."""
    print("=== Training Step Example ===\n")
    
    # Setup
    env = minatar.Environment('breakout')
    env.reset()
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    # Create network and agent
    network = DQN(action_dim=action_dim)
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-3
    )
    
    # Create dummy batch of transitions (normally from replay buffer)
    batch_size = 32
    dummy_observations = jnp.ones((batch_size,) + obs_shape)
    dummy_actions = jnp.array([i % action_dim for i in range(batch_size)])
    dummy_targets = jnp.ones(batch_size) * 0.5  # Target Q-values
    
    print(f"Batch observations shape: {dummy_observations.shape}")
    print(f"Batch actions shape: {dummy_actions.shape}")
    print(f"Batch targets shape: {dummy_targets.shape}\n")
    
    # Create train step function
    train_step = create_train_step_fn(network, agent.optimizer)
    
    # Perform training step
    new_params, new_opt_state, loss = train_step(
        agent.params,
        agent.opt_state,
        dummy_observations,
        dummy_actions,
        dummy_targets
    )
    
    print(f"Loss: {loss:.4f}")
    print("Training step completed successfully!\n")
    
    # Update agent parameters
    agent.params = new_params
    agent.opt_state = new_opt_state


def example_episode():
    """Run a complete episode using the DQN agent."""
    print("=== Complete Episode Example ===\n")
    
    # Setup
    env = minatar.Environment('breakout')
    env.seed(42)
    env.reset()
    
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    # Create agent
    network = DQN(action_dim=action_dim)
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        seed=42
    )
    
    # Run episode
    total_reward = 0
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        obs = env.state()
        
        # Select action with exploration
        action = agent.select_action(obs, epsilon=0.1)
        
        # Take action in environment
        reward, done = env.act(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"Episode finished after {steps} steps")
    print(f"Total reward: {total_reward}\n")


if __name__ == "__main__":
    example_basic_usage()
    example_dueling_dqn()
    example_training_step()
    example_episode()
    
    print("All examples completed successfully!")
