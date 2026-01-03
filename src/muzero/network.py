from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


class RepresentationNet(nn.Module):
    
    obs_channels: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (B, C, H, W)  (PyTorch-style)
        # Convert from NCHW to NHWC for Flax
        x = jnp.transpose(obs, (0, 2, 3, 1))
        
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME'
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME'
        )(x)
        x = nn.relu(x)


        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME'
        )(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  

        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        return x  # (B, hidden_dim)
       

class DynamicsNet(nn.Module):

    num_actions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        a = nn.Embed(num_embeddings=self.num_actions, features=self.hidden_dim)(action)
        x = jnp.concatenate([state, a], axis=-1)

        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        next_hidden = nn.Dense(features=self.hidden_dim)(x)

        r = nn.Dense(features=self.hidden_dim)(next_hidden)
        r = nn.relu(r)
        reward = nn.Dense(features=1)(r)
        reward = jnp.squeeze(reward, axis=-1)
        return (next_hidden, reward)
    

class PredictionNet(nn.Module):
    
    num_actions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self,hidden: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = nn.Dense(features=self.hidden_dim)(hidden)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        policy_logits = nn.Dense(features=self.num_actions)(x)
        
        v = nn.Dense(features=self.hidden_dim)(hidden)
        v = nn.relu(v)
        value = nn.Dense(features=1)(v)
        value = jnp.squeeze(value, axis=-1)
        return (policy_logits, value)


class MuZeroNet(nn.Module):

    obs_channels: int
    num_actions: int
    hidden_dim: int = 128

    def setup(self) -> None:
        self.repr = RepresentationNet(
            obs_channels=self.obs_channels,
            hidden_dim=self.hidden_dim
        )
        self.dyn = DynamicsNet(
            num_actions=self.num_actions,
            hidden_dim=self.hidden_dim
        )
        self.pred = PredictionNet(
            num_actions=self.num_actions,
            hidden_dim=self.hidden_dim
        )


    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        hidden = self.repr(obs)
        policy_logits, value = self.pred(hidden)
        return hidden, policy_logits, value
    
    def recurrent_inference(self, hidden: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        next_hidden, reward = self.dyn(hidden, action)
        policy_logits, value = self.pred(next_hidden)
        return next_hidden, reward, policy_logits, value
    
    def init_all(self, obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Initialize all submodules including dynamics network."""
        hidden = self.repr(obs)
        policy_logits, value = self.pred(hidden)
        _, _ = self.dyn(hidden, action)
        return hidden, policy_logits, value
