"""
JAX Neural Network for Deep Q-Learning (DQN)
Uses Flax for defining the network architecture and Optax for optimization.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
import optax


class DQN(nn.Module):
    """Deep Q-Network using Flax.
    
    A convolutional neural network designed for processing MinAtar observations
    and outputting Q-values for each action.
    
    Attributes:
        action_dim: Number of possible actions
        features: Sequence of feature dimensions for hidden layers
    """
    action_dim: int
    features: Sequence[int] = (128, 128)
    
    @nn.compact
    def __call__(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input observation, shape (batch, height, width, channels) or (height, width, channels)
        
        Returns:
            Q-values for each action, shape (batch, action_dim) or (action_dim,)
        """
        # Handle single observation (no batch dimension)
        single_obs = False
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)
            single_obs = True
        
        # Convolutional layers
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1)(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Fully connected layers
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        
        # Output layer - Q-values for each action
        x = nn.Dense(self.action_dim)(x)
        
        # Remove batch dimension if input was single observation
        if single_obs:
            x = jnp.squeeze(x, axis=0)
        
        return x


class DuelingDQN(nn.Module):
    """Dueling DQN architecture that separates value and advantage streams.
    
    This architecture explicitly separates the estimation of state value V(s)
    and action advantages A(s,a), which can lead to better learning.
    
    Attributes:
        action_dim: Number of possible actions
        features: Sequence of feature dimensions for hidden layers
    """
    action_dim: int
    features: Sequence[int] = (128, 128)
    
    @nn.compact
    def __call__(self, x):
        """Forward pass through the dueling network.
        
        Args:
            x: Input observation, shape (batch, height, width, channels) or (height, width, channels)
        
        Returns:
            Q-values for each action, shape (batch, action_dim) or (action_dim,)
        """
        # Handle single observation (no batch dimension)
        single_obs = False
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)
            single_obs = True
        
        # Shared convolutional layers
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1)(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Shared fully connected layer
        x = nn.Dense(self.features[0])(x)
        x = nn.relu(x)
        
        # Value stream
        value = nn.Dense(self.features[1])(x)
        value = nn.relu(value)
        value = nn.Dense(1)(value)  # Single value output
        
        # Advantage stream
        advantage = nn.Dense(self.features[1])(x)
        advantage = nn.relu(advantage)
        advantage = nn.Dense(self.action_dim)(advantage)
        
        # Combine value and advantage using the dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
        
        # Remove batch dimension if input was single observation
        if single_obs:
            q_values = jnp.squeeze(q_values, axis=0)
        
        return q_values


class DQNAgent:
    """DQN Agent that wraps the network with training utilities.
    
    Provides methods for network initialization, action selection,
    and training utilities.
    """
    
    def __init__(
        self,
        network: nn.Module,
        observation_shape: tuple,
        action_dim: int,
        learning_rate: float = 1e-4,
        seed: int = 0
    ):
        """Initialize the DQN agent.
        
        Args:
            network: Flax network module (DQN or DuelingDQN)
            observation_shape: Shape of observations (height, width, channels)
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            seed: Random seed for initialization
        """
        self.network = network
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.rng = jax.random.PRNGKey(seed)
        
        # Initialize network parameters
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_obs = jnp.ones((1,) + observation_shape)
        self.params = network.init(init_rng, dummy_obs)
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
    
    def select_action(self, observation, epsilon: float = 0.0):
        """Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            epsilon: Probability of selecting random action
        
        Returns:
            Selected action (int)
        """
        self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
        
        if jax.random.uniform(rng1) < epsilon:
            # Random action
            action = jax.random.randint(rng2, (), 0, self.action_dim)
        else:
            # Greedy action
            q_values = self.network.apply(self.params, observation)
            action = jnp.argmax(q_values)
        
        return int(action)
    
    def get_q_values(self, observation):
        """Get Q-values for an observation.
        
        Args:
            observation: Current observation
        
        Returns:
            Q-values for all actions
        """
        return self.network.apply(self.params, observation)


def create_dqn_loss_fn(network):
    """Create a loss function for DQN training.
    
    Args:
        network: Flax network module
    
    Returns:
        Loss function that takes (params, observations, actions, targets)
    """
    def loss_fn(params, observations, actions, targets):
        """Compute DQN loss (Mean Squared Error).
        
        Args:
            params: Network parameters
            observations: Batch of observations
            actions: Batch of actions taken
            targets: Target Q-values
        
        Returns:
            Mean squared error loss
        """
        # Get Q-values for all actions
        q_values = network.apply(params, observations)
        
        # Select Q-values for the taken actions
        # Use advanced indexing: q_values[range(batch_size), actions]
        batch_size = observations.shape[0]
        selected_q_values = q_values[jnp.arange(batch_size), actions]
        
        # Compute MSE loss
        loss = jnp.mean((selected_q_values - targets) ** 2)
        
        return loss
    
    return loss_fn


def create_train_step_fn(network, optimizer):
    """Create a JIT-compiled training step function.
    
    Args:
        network: Flax network module
        optimizer: Optax optimizer
    
    Returns:
        JIT-compiled training step function
    """
    def loss_fn(params, observations, actions, targets):
        """Compute DQN loss (Mean Squared Error)."""
        q_values = network.apply(params, observations)
        batch_size = observations.shape[0]
        selected_q_values = q_values[jnp.arange(batch_size), actions]
        loss = jnp.mean((selected_q_values - targets) ** 2)
        return loss
    
    @jax.jit
    def train_step(params, opt_state, observations, actions, targets):
        """Single training step with JIT compilation.
        
        Args:
            params: Network parameters
            opt_state: Optimizer state
            observations: Batch of observations
            actions: Batch of actions
            targets: Target Q-values
        
        Returns:
            Updated parameters, optimizer state, and loss value
        """
        loss, grads = jax.value_and_grad(loss_fn)(params, observations, actions, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    return train_step
