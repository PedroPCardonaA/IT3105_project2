from __future__ import annotations
from jax import numpy as jnp
import numpy as np

def stack_obs(steps, index, history_length: int) -> jnp.ndarray:
    obs0 = np.asarray(steps[index].obs)
    H, W, C = obs0.shape
    
    # Collect observations for each history step
    obs_list = []
    for h in range(history_length):
        t = index - (history_length - 1 - h)
        if t < 0:
            # Use zeros for time steps before the start
            obs_list.append(np.zeros((H, W, C), dtype=obs0.dtype))
        else:
            obs_list.append(np.asarray(steps[t].obs))
    
    # Stack along channel dimension: (H, W, C*history_length)
    stacked = np.concatenate(obs_list, axis=-1)
    
    # Reshape to (history_length, C, H, W)
    stacked = stacked.reshape(H, W, history_length, C)
    stacked = np.transpose(stacked, (2, 3, 0, 1))  # (history_length, C, H, W)
    return jnp.clip(jnp.asarray(stacked, dtype=jnp.float32), 0.0, 1.0)

def value_target(steps, index: int, discount: float, td_steps: int) -> float:
    """Compute n-step return target with value bootstrapping.
    
    Returns: sum_{t=0}^{n-1} gamma^t * r_{index+t} + gamma^n * V(s_{index+n})
    """
    value = 0.0
    discount_factor = 1.0
    for t in range(td_steps):
        td_index = index + t
        if td_index >= len(steps):
            break
        reward = steps[td_index].reward
        value += discount_factor * reward
        discount_factor *= discount
        if steps[td_index].done:
            # Episode ended, no bootstrap needed
            return value
    # Bootstrap with value estimate at index+td_steps
    bootstrap_index = index + td_steps
    if bootstrap_index < len(steps) and not steps[bootstrap_index - 1].done:
        value += discount_factor * steps[bootstrap_index].root_value
    return value
