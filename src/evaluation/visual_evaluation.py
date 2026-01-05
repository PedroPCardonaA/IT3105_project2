"""
Visual Evaluation Script for DQN and MuZero Agents
Uses MinAtar's GUI to display agent performance graphically.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import minatar
import jax
import jax.numpy as jnp
from src.nn.network import DQN, DuelingDQN, DQNAgent
from src.muzero.network import RepresentationNet, DynamicsNet, PredictionNet, MuZeroNet
from src.muzero.mcts import MuZeroMCTS, MCTSconfig

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def render_frame(env_state, env_name: str, info_text: str = ""):
    """Render a MinAtar frame as RGB image.
    
    Args:
        env_state: MinAtar environment state
        env_name: Name of the environment
        info_text: Optional text to display on frame
    
    Returns:
        RGB numpy array
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # MinAtar state is (H, W, C) - visualize it
    if env_state.shape[2] == 4:  # Standard MinAtar
        # Create RGB visualization
        rgb = np.zeros((*env_state.shape[:2], 3))
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ]
        for i in range(env_state.shape[2]):
            for c in range(3):
                rgb[:, :, c] += env_state[:, :, i] * colors[i][c]
        rgb = np.clip(rgb, 0, 1)
    else:
        # For stacked frames, just use first 4 channels
        rgb = np.zeros((*env_state.shape[:2], 3))
        colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        for i in range(min(4, env_state.shape[2])):
            for c in range(3):
                rgb[:, :, c] += env_state[:, :, i] * colors[i][c]
        rgb = np.clip(rgb, 0, 1)
    
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(f"{env_name.upper()} - {info_text}", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Convert to RGB array
    fig.canvas.draw()
    # Use buffer_rgba() and convert to RGB
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)
    # Convert RGBA to RGB
    image = image[:, :, :3]
    plt.close(fig)
    
    return image


def load_dqn_agent(checkpoint_path: str, env_name: str = "breakout", network_type: str = "dueling"):
    """Load a trained DQN agent from checkpoint.
    
    Args:
        checkpoint_path: Path to DQN checkpoint .pkl file
        env_name: MinAtar game name
        network_type: 'standard' or 'dueling'
    
    Returns:
        Loaded DQNAgent
    """
    # Load checkpoint with JAX compatibility handling
    import jax._src.core as jax_core
    original_init = jax_core.ShapedArray.__init__
    
    def patched_init(self, shape, dtype, weak_type=False, **kwargs):
        # Remove incompatible kwargs
        kwargs.pop('named_shape', None)
        original_init(self, shape, dtype, weak_type, **kwargs)
    
    jax_core.ShapedArray.__init__ = patched_init
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        jax_core.ShapedArray.__init__ = original_init
        raise ValueError(
            f"Failed to load checkpoint: {checkpoint_path}\n"
            f"Error: {e}\n"
            f"The checkpoint file may be corrupted or incomplete.\n"
            f"Try using a different checkpoint file."
        )
    finally:
        jax_core.ShapedArray.__init__ = original_init
    
    # Create environment to get observation shape
    env = minatar.Environment(env_name)
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    # Create network
    NetworkClass = DuelingDQN if network_type == 'dueling' else DQN
    network = NetworkClass(action_dim=action_dim, features=(128, 128))
    
    # Create agent and load parameters
    agent = DQNAgent(
        network=network,
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=1e-4,  # Not used for evaluation
        seed=42
    )
    
    agent.params = checkpoint['params']
    
    print(f"✓ DQN agent loaded from {checkpoint_path}")
    print(f"  Environment: {env_name}")
    print(f"  Network: {network_type.capitalize()} DQN")
    print(f"  Actions: {action_dim}")
    
    return agent


def load_muzero_agent(checkpoint_path: str, env_name: str = "breakout"):
    """Load a trained MuZero agent from checkpoint.
    
    Args:
        checkpoint_path: Path to MuZero checkpoint .pkl file
        env_name: MinAtar game name
    
    Returns:
        Tuple of (representation_params, dynamics_params, prediction_params, networks, config)
    """
    # Load checkpoint with JAX compatibility handling
    import jax._src.core as jax_core
    original_init = jax_core.ShapedArray.__init__
    
    def patched_init(self, shape, dtype, weak_type=False, **kwargs):
        # Remove incompatible kwargs
        kwargs.pop('named_shape', None)
        original_init(self, shape, dtype, weak_type, **kwargs)
    
    jax_core.ShapedArray.__init__ = patched_init
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        jax_core.ShapedArray.__init__ = original_init
        raise ValueError(
            f"Failed to load checkpoint: {checkpoint_path}\n"
            f"Error: {e}\n"
            f"The checkpoint file may be corrupted or incomplete.\n"
            f"Try using a different checkpoint file."
        )
    finally:
        jax_core.ShapedArray.__init__ = original_init
    
    # Extract parameters - handle different checkpoint formats
    if 'params' in checkpoint and isinstance(checkpoint['params'], dict):
        # New format: {'params': {'repr': ..., 'dyn': ..., 'pred': ...}}
        params = checkpoint['params']
        repr_params = params.get('repr', params.get('representation_params'))
        dyn_params = params.get('dyn', params.get('dynamics_params'))
        pred_params = params.get('pred', params.get('prediction_params'))
        config = checkpoint.get('config', {})
    else:
        # Old format: separate keys
        repr_params = checkpoint.get('representation_params')
        dyn_params = checkpoint.get('dynamics_params')
        pred_params = checkpoint.get('prediction_params')
        config = checkpoint.get('config', {})
    
    # Create environment to get dimensions
    env = minatar.Environment(env_name)
    obs_shape = env.state().shape
    action_dim = env.num_actions()
    
    # Get config values or infer from checkpoint
    hidden_dim = config.get('hidden_dim', 128)
    
    # Try to infer obs_channels and history_len from the saved parameters
    if repr_params and 'Conv_0' in repr_params and 'kernel' in repr_params['Conv_0']:
        # Infer from kernel shape: (height, width, in_channels, out_channels)
        obs_channels = repr_params['Conv_0']['kernel'].shape[2]
        single_obs_channels = obs_shape[2]  # Actual channels per frame
        history_len = obs_channels // single_obs_channels
        print(f"  Detected {obs_channels} input channels from checkpoint")
        print(f"  Using frame stacking: {history_len} frames × {single_obs_channels} channels")
    else:
        obs_channels = obs_shape[2]  # Last dimension is channels in MinAtar
        history_len = 1
    
    # Create networks
    repr_net = RepresentationNet(obs_channels=obs_channels, hidden_dim=hidden_dim)
    dyn_net = DynamicsNet(num_actions=action_dim, hidden_dim=hidden_dim)
    pred_net = PredictionNet(num_actions=action_dim, hidden_dim=hidden_dim)
    
    print(f"✓ MuZero agent loaded from {checkpoint_path}")
    print(f"  Environment: {env_name}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Actions: {action_dim}")
    
    return repr_params, dyn_params, pred_params, (repr_net, dyn_net, pred_net), config, history_len


def visualize_dqn_agent(agent: DQNAgent, env_name: str = "breakout", 
                        num_episodes: int = 3, epsilon: float = 0.0,
                        display_time: int = 50, seed: int = 42,
                        save_gif: bool = False, output_dir: str = "outputs"):
    """Visualize DQN agent playing with GUI.
    
    Args:
        agent: Trained DQNAgent
        env_name: MinAtar game name
        num_episodes: Number of episodes to play
        epsilon: Exploration rate (0.0 = fully greedy)
        display_time: Milliseconds to display each frame
        seed: Random seed
        save_gif: Whether to save gameplay as GIF
        output_dir: Directory to save GIFs
    """
    print(f"\n{'='*70}")
    print(f"DQN Agent Visual Evaluation")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Policy: {'Greedy (ε=0)' if epsilon == 0.0 else f'ε-greedy (ε={epsilon})'}")
    print(f"Display speed: {display_time}ms per frame")
    print(f"{'='*70}\n")
    
    rng = jax.random.PRNGKey(seed)
    env = minatar.Environment(env_name)
    
    # Create output directory if saving GIFs
    if save_gif:
        if not HAS_IMAGEIO:
            print("Warning: imageio not installed. Install with: uv pip install imageio")
            print("Continuing without GIF saving...\n")
            save_gif = False
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        env.reset()
        state = env.state()
        
        total_reward = 0
        steps = 0
        done = False
        frames = [] if save_gif else None
        
        # Capture initial state
        if save_gif:
            frame = render_frame(state, env_name, f"Step 0 | Reward: 0")
            frames.append(frame)
        else:
            # Display initial state
            env.display_state(time=display_time)
        
        while not done:
            # Select action
            rng, action_key = jax.random.split(rng)
            
            if jax.random.uniform(action_key) < epsilon:
                # Random action (exploration)
                action = int(jax.random.randint(action_key, (), 0, env.num_actions()))
            else:
                # Greedy action from agent (epsilon=0.0 for greedy)
                action = agent.select_action(state, epsilon=0.0)
            
            # Take action
            reward, done = env.act(action)
            state = env.state()
            
            total_reward += reward
            steps += 1
            
            # Capture or display state
            if save_gif:
                frame = render_frame(state, env_name, f"Step {steps} | Reward: {total_reward}")
                frames.append(frame)
            else:
                env.display_state(time=display_time)
        
        print(f"  Total Reward: {total_reward}")
        print(f"  Steps: {steps}")
        
        # Save GIF if requested
        if save_gif and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = Path(output_dir) / f"dqn_{env_name}_ep{episode}_{timestamp}.gif"
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
            print(f"  Saved GIF: {gif_path}")
    
    env.close_display()
    print(f"\n{'='*70}")
    print("DQN Visualization Complete!")
    print(f"{'='*70}\n")


def visualize_muzero_agent(repr_params, dyn_params, pred_params, networks,
                           config, history_len, env_name: str = "breakout",
                           num_episodes: int = 3, num_simulations: int = 50,
                           display_time: int = 50, seed: int = 42,
                           save_gif: bool = False, output_dir: str = "outputs"):
    """Visualize MuZero agent playing with GUI.
    
    Args:
        repr_params: Representation network parameters
        dyn_params: Dynamics network parameters
        pred_params: Prediction network parameters
        networks: Tuple of (repr_net, dyn_net, pred_net)
        config: MuZero configuration dict
        history_len: Number of frames to stack
        env_name: MinAtar game name
        num_episodes: Number of episodes to play
        num_simulations: MCTS simulations per action
        display_time: Milliseconds to display each frame
        seed: Random seed
        save_gif: Whether to save gameplay as GIF
        output_dir: Directory to save GIFs
    """
    print(f"\n{'='*70}")
    print(f"MuZero Agent Visual Evaluation")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"MCTS Simulations: {num_simulations}")
    print(f"Display speed: {display_time}ms per frame")
    print(f"{'='*70}\n")
    
    repr_net, dyn_net, pred_net = networks
    rng = jax.random.PRNGKey(seed)
    env = minatar.Environment(env_name)
    
    # Create output directory if saving GIFs
    if save_gif:
        if not HAS_IMAGEIO:
            print("Warning: imageio not installed. Install with: uv pip install imageio")
            print("Continuing without GIF saving...\n")
            save_gif = False
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a wrapper class for MuZero inference
    class MuZeroNetWrapper:
        def __init__(self, repr_net, dyn_net, pred_net, repr_params, dyn_params, pred_params):
            self.repr_net = repr_net
            self.dyn_net = dyn_net
            self.pred_net = pred_net
            self.repr_params = repr_params
            self.dyn_params = dyn_params
            self.pred_params = pred_params
        
        def initial_inference(self, obs):
            hidden = self.repr_net.apply({'params': self.repr_params}, obs)
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, hidden)
            return hidden, policy_logits, value
        
        def recurrent_inference(self, hidden, action):
            next_hidden, reward = self.dyn_net.apply({'params': self.dyn_params}, hidden[None, ...], action[None, ...])
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, next_hidden)
            return next_hidden[0], reward[0], policy_logits[0], value[0]
        
        def pred(self, hidden):
            policy_logits, value = self.pred_net.apply({'params': self.pred_params}, hidden[None, ...])
            return policy_logits[0], value[0]
    
    muzero_net = MuZeroNetWrapper(
        repr_net, dyn_net, pred_net,
        repr_params, dyn_params, pred_params
    )
    
    mcts_config = MCTSconfig(
        num_simulations=num_simulations,
        pb_c_base=config.get('pb_c_base', 19652),
        pb_c_init=config.get('c_puct', 1.25),
        dirichlet_alpha=config.get('dirichlet_alpha', 0.25),
        root_exploration_frac=0.0  # No exploration for evaluation
    )
    
    mcts = MuZeroMCTS(
        config=mcts_config,
        num_actions=env.num_actions(),
        network=muzero_net
    )
    
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        env.reset()
        obs = env.state()
        
        # Initialize frame history for stacking
        from collections import deque
        frame_history = deque(maxlen=history_len)
        for _ in range(history_len):
            frame_history.append(obs)
        
        total_reward = 0
        steps = 0
        done = False
        frames = [] if save_gif else None
        
        # Capture or display initial state
        if save_gif:
            frame = render_frame(obs, env_name, f"Step 0 | Reward: 0")
            frames.append(frame)
        else:
            env.display_state(time=display_time)
        
        while not done:
            # Stack frames: concatenate along channel dimension
            stacked_obs = jnp.concatenate(list(frame_history), axis=-1)
            
            # Prepare observation for network (NCHW format)
            obs_jax = jnp.array(stacked_obs, dtype=jnp.float32)
            # MinAtar gives (H, W, C), convert to (1, C, H, W) for network
            obs_batch = jnp.transpose(obs_jax, (2, 0, 1))[None, ...]
            
            # Get hidden state from representation network
            hidden = repr_net.apply({'params': repr_params}, obs_batch)
            
            # Run MCTS to get action
            policy, _ = mcts.run(hidden[0])  # Remove batch dimension
            
            # Select best action (greedy)
            action = int(jnp.argmax(policy))
            
            # Take action
            reward, done = env.act(action)
            obs = env.state()
            
            # Add new observation to history
            frame_history.append(obs)
            
            total_reward += reward
            steps += 1
            
            # Capture or display state
            if save_gif:
                frame = render_frame(obs, env_name, f"Step {steps} | Reward: {total_reward}")
                frames.append(frame)
            else:
                env.display_state(time=display_time)
        
        print(f"  Total Reward: {total_reward}")
        print(f"  Steps: {steps}")
        
        # Save GIF if requested
        if save_gif and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = Path(output_dir) / f"muzero_{env_name}_ep{episode}_{timestamp}.gif"
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
            print(f"  Saved GIF: {gif_path}")
    
    env.close_display()
    print(f"\n{'='*70}")
    print("MuZero Visualization Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visually evaluate DQN or MuZero agents using MinAtar GUI"
    )
    
    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['dqn', 'muzero'],
        required=True,
        help='Type of agent to evaluate'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pkl file)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='breakout',
        choices=['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders'],
        help='MinAtar environment name (default: breakout)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of episodes to visualize (default: 3)'
    )
    
    parser.add_argument(
        '--speed',
        type=int,
        default=50,
        help='Display speed in milliseconds per frame (default: 50)'
    )
    
    parser.add_argument(
        '--network-type',
        type=str,
        default='dueling',
        choices=['standard', 'dueling'],
        help='DQN network type (default: dueling, only for DQN agent)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0,
        help='Exploration rate for DQN (default: 0.0 = greedy, only for DQN agent)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='MCTS simulations per action (default: 50, only for MuZero agent)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--save-gif',
        action='store_true',
        help='Save gameplay as GIF instead of displaying (requires imageio)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save GIFs (default: outputs)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MinAtar Agent Visual Evaluation")
    print("="*70)
    
    if args.agent_type == 'dqn':
        # Load and visualize DQN agent
        agent = load_dqn_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            network_type=args.network_type
        )
        
        visualize_dqn_agent(
            agent=agent,
            env_name=args.env,
            num_episodes=args.episodes,
            epsilon=args.epsilon,
            display_time=args.speed,
            seed=args.seed,
            save_gif=args.save_gif,
            output_dir=args.output_dir
        )
    
    elif args.agent_type == 'muzero':
        # Load and visualize MuZero agent
        repr_params, dyn_params, pred_params, networks, config, history_len = load_muzero_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env
        )
        
        visualize_muzero_agent(
            repr_params=repr_params,
            dyn_params=dyn_params,
            pred_params=pred_params,
            networks=networks,
            config=config,
            history_len=history_len,
            env_name=args.env,
            num_episodes=args.episodes,
            num_simulations=args.simulations,
            display_time=args.speed,
            seed=args.seed,
            save_gif=args.save_gif,
            output_dir=args.output_dir
        )
    
    print("Done! Close the display window to exit.")


if __name__ == '__main__':
    main()
