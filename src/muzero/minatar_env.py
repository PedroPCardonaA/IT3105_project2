from __future__ import annotations
from typing import Tuple
import jax.numpy as jnp

class MinatarEnv:
    
    def __init__(self, game = "breakout", seed: int = 42):
        import minatar
        self.env = minatar.Environment(game)
        self.env.seed(seed)
        self._obs = None

    def reset(self) -> jnp.ndarray:
        self.env.reset()
        self._obs = jnp.array(self.env.state(), dtype=jnp.float32)
        return self._obs
    
    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool]:
        reward, done = self.env.act(action)
        self._obs = jnp.array(self.env.state(), dtype=jnp.float32)
        return self._obs, reward, done
    
    def num_actions(self) -> int:
        return int(self.env.num_actions())
    
    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        if self._obs is None:
            _ = self.reset()
        assert self._obs is not None
        shape = tuple(self._obs.shape)
        assert len(shape) == 3
        return shape