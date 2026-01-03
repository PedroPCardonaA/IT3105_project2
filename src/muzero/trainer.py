from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from collections import defaultdict

from network import MuZeroNet
from buffer import EpisodeBuffer, Episode
from utils import stack_obs, value_target


@dataclass(frozen=True)
class TrainConfig:
    history_len: int = 8      # q+1 (look-back window)
    unroll_steps: int = 5     # w (roll-ahead actions)
    td_steps: int = 10
    discount: float = 0.997
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0


class TrainState(train_state.TrainState):
    """Extended train state with additional fields if needed."""
    pass


class MuZeroTrainer:
    """
    Composite training (BPTT-like unroll) over Ψ=(NNr,NNd,NNp) as in the PDF.
    JAX/Flax implementation.
    """
    def __init__(
        self, 
        network: MuZeroNet, 
        params, 
        num_actions: int, 
        obs_shape_hwc: Tuple[int, int, int], 
        cfg: TrainConfig
    ):
        self.net = network
        self.num_actions = num_actions
        self.obs_shape_hwc = obs_shape_hwc
        self.cfg = cfg

        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
        )
        
        # Create train state
        self.state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optimizer
        )
        
        # Initialize metrics tracking for visualization
        self.training_step = 0
        self.metrics_history = defaultdict(list)

    def sample_batch(self, buffer: EpisodeBuffer):
        """Sample batch from buffer matching PyTorch implementation."""
        # Randomly sample episodes
        import random
        batch = []
        for _ in range(self.cfg.batch_size):
            episode = random.choice(buffer.episodes)
            start = random.randint(0, len(episode) - 1)
            batch.append((episode, start))
        return batch

    def train_batch(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        batch = self.sample_batch(buffer)

        obs_batch = []
        actions_batch = []
        policy_targets = []
        value_targets = []
        reward_targets = []
        mask_state = []
        mask_reward = []

        for ep, start in batch:
            steps = ep.steps
            stacked = stack_obs(steps, start, self.cfg.history_len)  # (hist, C, H, W)
            obs_batch.append(stacked)

            ep_actions = []
            ep_pi = []
            ep_v = []
            ep_r = []
            ep_ms = []
            ep_mr = []

            for i in range(self.cfg.unroll_steps + 1):
                t = start + i
                valid_state = (t < len(steps)) and (i == 0 or not steps[t - 1].done)
                ep_ms.append(1.0 if valid_state else 0.0)

                if valid_state:
                    ep_pi.append(steps[t].policy)
                    ep_v.append(value_target(steps, t, self.cfg.discount, self.cfg.td_steps))
                else:
                    ep_pi.append(jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions)
                    ep_v.append(0.0)

                if i < self.cfg.unroll_steps:
                    valid_reward = (t < len(steps)) and not (i > 0 and steps[t - 1].done)
                    ep_mr.append(1.0 if valid_reward else 0.0)
                    if valid_reward:
                        ep_actions.append(int(steps[t].action))
                        ep_r.append(float(steps[t].reward))
                    else:
                        ep_actions.append(0)
                        ep_r.append(0.0)

            actions_batch.append(ep_actions)
            policy_targets.append(jnp.stack(ep_pi))
            value_targets.append(jnp.array(ep_v, dtype=jnp.float32))
            reward_targets.append(jnp.array(ep_r, dtype=jnp.float32))
            mask_state.append(jnp.array(ep_ms, dtype=jnp.float32))
            mask_reward.append(jnp.array(ep_mr, dtype=jnp.float32))

        # Stack into batched arrays
        obs = jnp.stack(obs_batch)  # (B, hist, C, H, W)
        # Reshape to (B, hist*C, H, W) for network input
        B, hist, C, H, W = obs.shape
        obs = obs.reshape(B, hist * C, H, W)
        
        actions = jnp.array(actions_batch, dtype=jnp.int32)  # (B, unroll_steps)
        pi_t = jnp.stack(policy_targets)  # (B, unroll_steps+1, num_actions)
        v_t = jnp.stack(value_targets)  # (B, unroll_steps+1)
        r_t = jnp.stack(reward_targets)  # (B, unroll_steps)
        ms = jnp.stack(mask_state)  # (B, unroll_steps+1)
        mr = jnp.stack(mask_reward)  # (B, unroll_steps)

        # Define loss function
        def loss_fn(params):
            # Initial representation
            hidden = self.net.apply(params, obs, method=lambda net, x: net.repr(x))

            pol_loss = 0.0
            val_loss = 0.0
            rew_loss = 0.0

            # Step 0 prediction
            logits, value = self.net.apply(params, hidden, method=lambda net, h: net.pred(h))
            pol_loss += self._policy_loss(logits, pi_t[:, 0], ms[:, 0])
            val_loss += self._mse(value, v_t[:, 0], ms[:, 0])

            # Unroll
            for i in range(self.cfg.unroll_steps):
                hidden, r_pred = self.net.apply(
                    params, hidden, actions[:, i], 
                    method=lambda net, h, a: net.dyn(h, a)
                )
                rew_loss += self._mse(r_pred, r_t[:, i], mr[:, i])

                logits, value = self.net.apply(params, hidden, method=lambda net, h: net.pred(h))
                pol_loss += self._policy_loss(logits, pi_t[:, i + 1], ms[:, i + 1])
                val_loss += self._mse(value, v_t[:, i + 1], ms[:, i + 1])

            total_loss = (
                self.cfg.policy_loss_weight * pol_loss
                + self.cfg.value_loss_weight * val_loss
                + self.cfg.reward_loss_weight * rew_loss
            )

            return total_loss, {
                "loss": total_loss,
                "policy_loss": pol_loss,
                "value_loss": val_loss,
                "reward_loss": rew_loss,
            }

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(self.state.params)

        # Update parameters
        self.state = self.state.apply_gradients(grads=grads)

        # Convert metrics to Python floats
        metrics_float = {k: float(v) for k, v in metrics.items()}
        
        # Record metrics for visualization
        self.training_step += 1
        for k, v in metrics_float.items():
            self.metrics_history[k].append(v)
        
        return metrics_float

    def _policy_loss(self, logits: jnp.ndarray, target_pi: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Compute cross-entropy loss for policy."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -(target_pi * log_probs).sum(axis=-1)  # (B,)
        return (ce * mask).sum() / (mask.sum() + 1e-8)

    def _mse(self, pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Compute masked mean squared error."""
        mse = (pred - target) ** 2
        return (mse * mask).sum() / (mask.sum() + 1e-8)
    
    def plot_training_progress(self, save_path: str = None, show: bool = True, window_size: int = 100):
        """Plot training metrics over time.
        
        Args:
            save_path: Path to save the plot. If None, plot is not saved.
            show: Whether to display the plot.
            window_size: Window size for computing moving average (smoothing).
        """
        if not self.metrics_history:
            print("No training metrics to plot yet.")
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MuZero Training Progress', fontsize=16, fontweight='bold')
        
        steps = list(range(1, self.training_step + 1))
        
        # Plot total loss
        if 'loss' in self.metrics_history:
            ax = axes[0, 0]
            loss_values = self.metrics_history['loss']
            ax.plot(steps, loss_values, alpha=0.3, color='blue', label='Raw')
            if len(loss_values) >= window_size:
                smoothed = self._moving_average(loss_values, window_size)
                ax.plot(steps[window_size-1:], smoothed, color='blue', linewidth=2, label=f'MA({window_size})')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Total Loss')
            ax.set_title('Total Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot policy loss
        if 'policy_loss' in self.metrics_history:
            ax = axes[0, 1]
            policy_values = self.metrics_history['policy_loss']
            ax.plot(steps, policy_values, alpha=0.3, color='green', label='Raw')
            if len(policy_values) >= window_size:
                smoothed = self._moving_average(policy_values, window_size)
                ax.plot(steps[window_size-1:], smoothed, color='green', linewidth=2, label=f'MA({window_size})')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Policy Loss')
            ax.set_title('Policy Loss (Cross-Entropy)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot value loss
        if 'value_loss' in self.metrics_history:
            ax = axes[1, 0]
            value_values = self.metrics_history['value_loss']
            ax.plot(steps, value_values, alpha=0.3, color='red', label='Raw')
            if len(value_values) >= window_size:
                smoothed = self._moving_average(value_values, window_size)
                ax.plot(steps[window_size-1:], smoothed, color='red', linewidth=2, label=f'MA({window_size})')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value Loss')
            ax.set_title('Value Loss (MSE)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot reward loss
        if 'reward_loss' in self.metrics_history:
            ax = axes[1, 1]
            reward_values = self.metrics_history['reward_loss']
            ax.plot(steps, reward_values, alpha=0.3, color='orange', label='Raw')
            if len(reward_values) >= window_size:
                smoothed = self._moving_average(reward_values, window_size)
                ax.plot(steps[window_size-1:], smoothed, color='orange', linewidth=2, label=f'MA({window_size})')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Reward Loss')
            ax.set_title('Reward Loss (MSE)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training progress plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_comparison(self, save_path: str = None, show: bool = True):
        """Plot all loss components on the same graph for comparison.
        
        Args:
            save_path: Path to save the plot. If None, plot is not saved.
            show: Whether to display the plot.
        """
        if not self.metrics_history:
            print("No training metrics to plot yet.")
            return
        
        plt.figure(figsize=(12, 6))
        steps = list(range(1, self.training_step + 1))
        
        colors = {'policy_loss': 'green', 'value_loss': 'red', 'reward_loss': 'orange'}
        
        for metric_name in ['policy_loss', 'value_loss', 'reward_loss']:
            if metric_name in self.metrics_history:
                values = self.metrics_history[metric_name]
                label = metric_name.replace('_', ' ').title()
                plt.plot(steps, values, alpha=0.6, color=colors[metric_name], label=label)
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title('MuZero Loss Components Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss comparison plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _moving_average(self, values: List[float], window_size: int) -> List[float]:
        """Compute moving average for smoothing plots.
        
        Args:
            values: List of values to smooth.
            window_size: Size of the moving window.
            
        Returns:
            List of smoothed values.
        """
        smoothed = []
        for i in range(window_size - 1, len(values)):
            window = values[i - window_size + 1:i + 1]
            smoothed.append(sum(window) / window_size)
        return smoothed
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics.
        
        Returns:
            Dictionary with the latest metric values.
        """
        if not self.metrics_history:
            return {}
        return {k: v[-1] for k, v in self.metrics_history.items()}
    
    def print_training_summary(self):
        """Print a summary of training progress."""
        if not self.metrics_history:
            print("No training has been performed yet.")
            return
        
        print("\n" + "="*60)
        print(f"MuZero Training Summary (Step {self.training_step})")
        print("="*60)
        
        for metric_name, values in self.metrics_history.items():
            if values:
                latest = values[-1]
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                
                print(f"\n{metric_name.replace('_', ' ').title()}:")
                print(f"  Current: {latest:.6f}")
                print(f"  Min:     {min_val:.6f}")
                print(f"  Max:     {max_val:.6f}")
                print(f"  Average: {avg_val:.6f}")
        
        print("\n" + "="*60 + "\n")

