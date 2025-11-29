"""
Play and record video for Custom PPO trained models with velocity arrows.

Usage:
    ./isaaclab.bat -p scripts/play_video.py --task Isaac-MyAnymal-Flat-v0 --num_envs 64 --checkpoint logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37/model_10000.pt
"""

import argparse
import os
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Play and record Custom PPO model")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--video_length", type=int, default=500, help="Video length in frames")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Enable cameras for video recording
args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class EmpiricalNormalization:
    """Observation normalization using Welford's algorithm."""

    def __init__(self, shape, eps=1e-8, device="cuda:0"):
        self.eps = eps
        self.device = device
        self.running_mean = torch.zeros(shape, device=device)
        self.running_var = torch.ones(shape, device=device)
        self.count = 0

    def normalize(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

    def load_state_dict(self, state_dict):
        self.running_mean = state_dict["running_mean"].to(self.device)
        self.running_var = state_dict["running_var"].to(self.device)
        self.count = state_dict["count"]


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network matching training architecture."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[128, 128, 128]):
        super().__init__()

        # Actor network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU()
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, obs):
        """Get deterministic action (mean)."""
        return self.actor(obs)


def main():
    device = "cuda:0"

    # Parse environment config
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    # Create environment WITH render_mode for video recording
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    print(f"\n{'='*60}")
    print(f"Custom PPO Video Recorder")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Environments: {args.num_envs}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Video length: {args.video_length} frames")
    print(f"{'='*60}\n")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Detect network architecture from checkpoint
    hidden_dims = []
    layer_idx = 0
    while f"actor.{layer_idx}.weight" in checkpoint["model_state_dict"]:
        weight = checkpoint["model_state_dict"][f"actor.{layer_idx}.weight"]
        hidden_dims.append(weight.shape[0])
        layer_idx += 2
    hidden_dims = hidden_dims[:-1]

    print(f"[INFO] Detected hidden dims: {hidden_dims}")

    # Create network
    network = ActorCriticNetwork(obs_dim, action_dim, hidden_dims).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    print("[INFO] Model loaded successfully!")

    # Load observation normalizer
    obs_normalizer = EmpiricalNormalization(obs_dim, device=device)
    if "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        print("[INFO] Observation normalizer loaded!")
    else:
        print("[WARNING] No observation normalizer in checkpoint!")

    # Setup video directory
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    video_dir = os.path.join(checkpoint_dir, "videos", "custom_play")
    os.makedirs(video_dir, exist_ok=True)

    print(f"[INFO] Video will be saved to: {video_dir}")

    # Wrap environment for video recording (same as RSL-RL play.py)
    video_kwargs = {
        "video_folder": video_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Run simulation
    print(f"\n[INFO] Starting recording ({args.video_length} frames)...")
    print("[INFO] Velocity arrows: Red=Command, Green=Actual, Cyan=Heading")

    obs_dict, _ = env.reset()
    # Isaac Lab returns obs as dict with 'policy' key
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    episode_rewards = []
    current_rewards = torch.zeros(args.num_envs, device=device)

    for step in range(args.video_length):
        # Normalize observations
        obs_normalized = obs_normalizer.normalize(obs)

        # Get action (deterministic)
        with torch.no_grad():
            actions = network.act(obs_normalized)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        current_rewards += rewards

        # Track episode completions
        dones = terminated | truncated
        if dones.any():
            episode_rewards.extend(current_rewards[dones].cpu().tolist())
            current_rewards[dones] = 0

        # Progress update
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{args.video_length}")

    # Close environment (saves video)
    env.close()

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Recording Complete!")
    print(f"{'='*60}")
    print(f"Total steps: {args.video_length}")
    if episode_rewards:
        print(f"Completed episodes: {len(episode_rewards)}")
        print(f"Mean episode reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"\nVideo saved to: {video_dir}")
    print(f"{'='*60}\n")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()