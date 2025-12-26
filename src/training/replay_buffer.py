"""
Replay Buffer for DQN training
"""

import numpy as np
from collections import deque
import jax.numpy as jnp


class ReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores transitions and provides random sampling for training.
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as JAX arrays
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards),
            jnp.array(next_states),
            jnp.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
