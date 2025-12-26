# JAX Neural Networks for Deep Q-Learning (DQN)

This module provides JAX/Flax implementations of neural networks for Deep Q-Learning, optimized for MinAtar environments.

## Features

- **DQN Network**: Standard Deep Q-Network with convolutional layers
- **Dueling DQN**: Dueling architecture that separates value and advantage streams
- **DQN Agent**: Complete agent with action selection and training utilities
- **JIT-compiled Training**: Fast training steps using JAX's JIT compilation
- **Epsilon-greedy Exploration**: Built-in exploration strategy

## Installation

Required packages (already in pyproject.toml):
```bash
pip install jax flax optax gymnasium minatar
```

## Quick Start

### Basic DQN Usage

```python
import minatar
from src.nn.network import DQN, DQNAgent

# Create environment
env = minatar.Environment('breakout')
env.reset()

# Setup network
obs_shape = env.state().shape  # (10, 10, 4)
action_dim = env.num_actions()  # 6

network = DQN(action_dim=action_dim, features=(128, 128))
agent = DQNAgent(
    network=network,
    observation_shape=obs_shape,
    action_dim=action_dim,
    learning_rate=1e-4,
    seed=42
)

# Get Q-values
obs = env.state()
q_values = agent.get_q_values(obs)

# Select action
action = agent.select_action(obs, epsilon=0.1)  # epsilon-greedy
```

### Dueling DQN

```python
from src.nn.network import DuelingDQN

# Use Dueling architecture instead
network = DuelingDQN(action_dim=action_dim, features=(128, 128))
agent = DQNAgent(network=network, ...)
```

### Training Step

```python
from src.nn.network import create_train_step_fn
import jax.numpy as jnp

# Create training function
train_step = create_train_step_fn(network, agent.optimizer)

# Prepare batch (from replay buffer)
batch_obs = jnp.array([...])  # (batch_size, 10, 10, 4)
batch_actions = jnp.array([...])  # (batch_size,)
batch_targets = jnp.array([...])  # (batch_size,)

# Train
new_params, new_opt_state, loss = train_step(
    agent.params,
    agent.opt_state,
    batch_obs,
    batch_actions,
    batch_targets
)

# Update agent
agent.params = new_params
agent.opt_state = new_opt_state
```

## Network Architectures

### DQN
```
Input (10, 10, 4)
    ↓
Conv2D(16, 3x3) + ReLU
    ↓
Conv2D(32, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense(128) + ReLU
    ↓
Dense(128) + ReLU
    ↓
Dense(action_dim)
    ↓
Q-values (action_dim,)
```

### Dueling DQN
```
Input (10, 10, 4)
    ↓
Conv2D(16, 3x3) + ReLU
    ↓
Conv2D(32, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense(128) + ReLU
    ↓
    ├─────────────┬─────────────┐
    ↓             ↓             ↓
Value Stream   Advantage Stream
Dense(128)     Dense(128)
    ↓             ↓
Dense(1)       Dense(action_dim)
    ↓             ↓
    └─────────────┴─────────────┘
              ↓
    Q = V + (A - mean(A))
              ↓
        Q-values (action_dim,)
```

## API Reference

### DQN(action_dim, features=(128, 128))
Standard DQN network.

**Parameters:**
- `action_dim` (int): Number of possible actions
- `features` (Sequence[int]): Hidden layer dimensions

### DuelingDQN(action_dim, features=(128, 128))
Dueling DQN architecture.

**Parameters:**
- `action_dim` (int): Number of possible actions
- `features` (Sequence[int]): Hidden layer dimensions for value/advantage streams

### DQNAgent(network, observation_shape, action_dim, learning_rate, seed)
Agent wrapper with training utilities.

**Parameters:**
- `network` (nn.Module): Flax network (DQN or DuelingDQN)
- `observation_shape` (tuple): Shape of observations
- `action_dim` (int): Number of actions
- `learning_rate` (float): Learning rate for Adam optimizer
- `seed` (int): Random seed

**Methods:**
- `select_action(observation, epsilon=0.0)`: Select action using epsilon-greedy policy
- `get_q_values(observation)`: Get Q-values for an observation

### create_train_step_fn(network, optimizer)
Create JIT-compiled training function.

**Parameters:**
- `network` (nn.Module): Flax network
- `optimizer` (optax.GradientTransformation): Optimizer

**Returns:**
- Function: `train_step(params, opt_state, observations, actions, targets)`

## Complete Example

See [examples/dqn_example.py](../examples/dqn_example.py) for complete usage examples including:
- Basic DQN usage
- Dueling DQN usage
- Training steps
- Running complete episodes

## Tips for DQN Training

1. **Use a replay buffer** to store and sample experiences
2. **Target network**: Maintain a separate target network for stability
3. **Batch size**: Typically 32-64 for MinAtar
4. **Learning rate**: Start with 1e-4 or 1e-3
5. **Epsilon decay**: Start at 1.0, decay to 0.01-0.1
6. **Update frequency**: Update every 4 steps is common
7. **Target update**: Update target network every 1000-10000 steps

## Performance Notes

- All forward passes and training steps are JIT-compiled for speed
- Supports both single observations and batched inputs
- Gradient computation is automatic via JAX
- Compatible with standard DQN algorithms and variants (Double DQN, etc.)
