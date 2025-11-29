"""
Keyboard Control for Anymal-C in Isaac Lab
===========================================

Based on official Isaac Sim Spot Quadruped Example.

Controls (both WASD and Arrow/Numpad work):
    W / UP / NUM8      - Forward
    S / DOWN / NUM2    - Backward
    A / LEFT / NUM4    - Strafe Left
    D / RIGHT / NUM6   - Strafe Right
    Q / N / NUM7       - Turn Left (yaw+)
    E / M / NUM9       - Turn Right (yaw-)
    R                  - Reset robot
    ESC                - Quit

Usage:
    ./isaaclab.bat -p scripts/play_keyboard.py --task Isaac-MyAnymal-Flat-v0 --checkpoint logs/custom_ppo/anymal_custom/<timestamp>/model_best.pt

    For RSL-RL trained models:
    ./isaaclab.bat -p scripts/play_keyboard.py --task Isaac-MyAnymal-Flat-v0 --checkpoint logs/rsl_rl/anymal_c_flat_direct/<timestamp>/model_4999.pt
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard control for quadruped robot")
parser.add_argument("--task", type=str, default="Isaac-MyAnymal-Flat-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1  # Force single env for keyboard control

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import carb
import carb.input
import omni.appwindow


# =============================================================================
# OBSERVATION NORMALIZATION (Training ile aynı)
# =============================================================================

class EmpiricalNormalization(nn.Module):
    """Online observation normalization - must match training."""

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# =============================================================================
# SIMPLE ACTOR-CRITIC (RSL-RL checkpoint compatible)
# =============================================================================

def detect_hidden_dims_from_checkpoint(state_dict, num_obs):
    """Detect hidden dimensions from checkpoint state dict."""
    hidden_dims = []

    # Look for actor.0.weight, actor.2.weight, etc.
    layer_idx = 0
    while True:
        key = f"actor.{layer_idx}.weight"
        if key in state_dict:
            weight = state_dict[key]
            hidden_dims.append(weight.shape[0])
            layer_idx += 2  # Skip activation layer
        else:
            break

    # Remove last element (output layer)
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]

    return hidden_dims if hidden_dims else [128, 128, 128]  # RSL-RL default


class SimpleActorCritic(nn.Module):
    """Simple Actor-Critic for loading RSL-RL or custom checkpoints."""

    def __init__(self, num_obs, num_actions, hidden_dims=[128, 128, 128]):
        super().__init__()

        self.hidden_dims = hidden_dims

        # Actor MLP
        actor_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable std
        self.std = nn.Parameter(torch.ones(num_actions))
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def act_inference(self, obs):
        """Get deterministic action for inference."""
        return self.actor(obs)


# =============================================================================
# KEYBOARD CONTROLLER (Spot example style)
# =============================================================================

class QuadrupedKeyboardController:
    """
    Keyboard controller following official Spot example pattern.
    Uses smooth ramping for velocity changes.
    """

    def __init__(self):
        self._base_command = np.array([0.0, 0.0, 0.0])  # [vx, vy, yaw_rate]
        self._target_command = np.array([0.0, 0.0, 0.0])  # Target velocity
        self._current_command = np.array([0.0, 0.0, 0.0])  # Smoothed velocity

        # Velocity limits - Training range ile uyumlu
        # Isaac Lab Anymal-C default: lin_vel_x=(-1,1), lin_vel_y=(-1,1), ang_vel_z=(-1,1)
        self._max_lin_vel = 0.5   # m/s - Conservative for stability
        self._max_ang_vel = 0.5   # rad/s - Conservative for stability

        # Smoothing parameters - Yumuşak geçiş için
        self._ramp_rate = 0.05  # Velocity change per step (lower = smoother)

        # Key bindings: key_name -> [vx_delta, vy_delta, yaw_delta]
        # Multiple keys can map to same command
        self._input_keyboard_mapping = {
            # Forward (W, UP, NUMPAD_8)
            "W": [self._max_lin_vel, 0.0, 0.0],
            "UP": [self._max_lin_vel, 0.0, 0.0],
            "NUMPAD_8": [self._max_lin_vel, 0.0, 0.0],

            # Backward (S, DOWN, NUMPAD_2)
            "S": [-self._max_lin_vel, 0.0, 0.0],
            "DOWN": [-self._max_lin_vel, 0.0, 0.0],
            "NUMPAD_2": [-self._max_lin_vel, 0.0, 0.0],

            # Strafe Left (A, LEFT, NUMPAD_4)
            "A": [0.0, self._max_lin_vel, 0.0],
            "LEFT": [0.0, self._max_lin_vel, 0.0],
            "NUMPAD_4": [0.0, self._max_lin_vel, 0.0],

            # Strafe Right (D, RIGHT, NUMPAD_6)
            "D": [0.0, -self._max_lin_vel, 0.0],
            "RIGHT": [0.0, -self._max_lin_vel, 0.0],
            "NUMPAD_6": [0.0, -self._max_lin_vel, 0.0],

            # Turn Left / Yaw+ (Q, N, NUMPAD_7)
            "Q": [0.0, 0.0, self._max_ang_vel],
            "N": [0.0, 0.0, self._max_ang_vel],
            "NUMPAD_7": [0.0, 0.0, self._max_ang_vel],

            # Turn Right / Yaw- (E, M, NUMPAD_9)
            "E": [0.0, 0.0, -self._max_ang_vel],
            "M": [0.0, 0.0, -self._max_ang_vel],
            "NUMPAD_9": [0.0, 0.0, -self._max_ang_vel],
        }

        # Active keys tracking
        self._active_keys = set()

        # Flags
        self._reset_requested = False
        self._quit_requested = False

        # Setup keyboard input
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        print("[KEYBOARD] Controller initialized (smooth ramping enabled)")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Handle keyboard events - track active keys."""

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            if key_name in self._input_keyboard_mapping:
                self._active_keys.add(key_name)
            elif key_name == "R":
                self._reset_requested = True
            elif key_name == "ESCAPE":
                self._quit_requested = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key_name = event.input.name
            if key_name in self._active_keys:
                self._active_keys.discard(key_name)

        return True

    def get_command(self, device) -> torch.Tensor:
        """Get current velocity command with smooth ramping."""
        # Calculate target from active keys
        self._target_command = np.array([0.0, 0.0, 0.0])
        for key in self._active_keys:
            if key in self._input_keyboard_mapping:
                self._target_command += np.array(self._input_keyboard_mapping[key])

        # Clamp target to limits
        self._target_command = np.clip(
            self._target_command,
            [-self._max_lin_vel, -self._max_lin_vel, -self._max_ang_vel],
            [self._max_lin_vel, self._max_lin_vel, self._max_ang_vel]
        )

        # Smooth ramping towards target
        diff = self._target_command - self._current_command
        self._current_command += np.clip(diff, -self._ramp_rate, self._ramp_rate)

        return torch.tensor([self._current_command], device=device, dtype=torch.float32)

    def reset_command(self):
        """Reset velocity command to zero."""
        self._base_command = np.array([0.0, 0.0, 0.0])
        self._target_command = np.array([0.0, 0.0, 0.0])
        self._current_command = np.array([0.0, 0.0, 0.0])
        self._active_keys.clear()

    @property
    def reset_requested(self) -> bool:
        flag = self._reset_requested
        self._reset_requested = False
        return flag

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def print_status(self):
        """Print current command status."""
        vx, vy, yaw = self._current_command
        tx, ty, tyaw = self._target_command
        print(f"\r[CMD] Vx: {vx:+.2f}/{tx:+.2f} | Vy: {vy:+.2f}/{ty:+.2f} | Yaw: {yaw:+.2f}/{tyaw:+.2f}    ", end="", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    # Extend episode length for play mode
    env_cfg.episode_length_s = 1000.0

    # Disable observation noise for cleaner control
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")

    # Get device from unwrapped environment
    unwrapped_env = env.unwrapped
    device = unwrapped_env.device

    # Get dimensions
    num_obs = env.observation_space.shape[1]
    num_actions = env.action_space.shape[1]

    print(f"[INFO] Observation dim: {num_obs}")
    print(f"[INFO] Action dim: {num_actions}")
    print(f"[INFO] Device: {device}")

    # =========================================================================
    # MODEL SETUP
    # =========================================================================

    # Check if checkpoint exists
    checkpoint_path = args_cli.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints in logs/custom_ppo:")
        custom_ppo_dir = "logs/custom_ppo"
        if os.path.exists(custom_ppo_dir):
            for exp in os.listdir(custom_ppo_dir):
                exp_path = os.path.join(custom_ppo_dir, exp)
                if os.path.isdir(exp_path):
                    for run in os.listdir(exp_path):
                        run_path = os.path.join(exp_path, run)
                        if os.path.isdir(run_path):
                            checkpoints = [f for f in os.listdir(run_path) if f.endswith('.pt')]
                            if checkpoints:
                                print(f"  {run_path}/")
                                for ckpt in checkpoints:
                                    print(f"    - {ckpt}")
        print("\nAvailable checkpoints in logs/rsl_rl:")
        rsl_rl_dir = "logs/rsl_rl"
        if os.path.exists(rsl_rl_dir):
            for exp in os.listdir(rsl_rl_dir):
                exp_path = os.path.join(rsl_rl_dir, exp)
                if os.path.isdir(exp_path):
                    for run in os.listdir(exp_path):
                        run_path = os.path.join(exp_path, run)
                        if os.path.isdir(run_path):
                            checkpoints = [f for f in os.listdir(run_path) if f.endswith('.pt')]
                            if checkpoints:
                                print(f"  {run_path}/")
                                for ckpt in checkpoints[:3]:  # Show first 3
                                    print(f"    - {ckpt}")
                                if len(checkpoints) > 3:
                                    print(f"    ... and {len(checkpoints)-3} more")
        env.close()
        return

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        # Custom PPO format
        state_dict = checkpoint["model_state_dict"]
        has_obs_normalizer = "obs_normalizer" in checkpoint
    elif "actor_state_dict" in checkpoint:
        # Some RSL-RL formats
        state_dict = checkpoint
        has_obs_normalizer = False
    else:
        # Direct state dict (RSL-RL)
        state_dict = checkpoint
        has_obs_normalizer = False

    # Detect hidden dimensions from checkpoint
    hidden_dims = detect_hidden_dims_from_checkpoint(state_dict, num_obs)
    print(f"[INFO] Detected hidden dims: {hidden_dims}")

    # Create model with correct architecture
    actor_critic = SimpleActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
    ).to(device)

    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.eval()
    print("[INFO] Model loaded successfully!")

    # Load observation normalizer if available (CRITICAL for custom PPO!)
    obs_normalizer = None
    if has_obs_normalizer:
        obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        print("[INFO] Observation normalizer loaded!")
        print(f"       Running mean range: [{obs_normalizer.running_mean.min():.3f}, {obs_normalizer.running_mean.max():.3f}]")
        print(f"       Running var range:  [{obs_normalizer.running_var.min():.3f}, {obs_normalizer.running_var.max():.3f}]")
    else:
        print("[INFO] No observation normalizer in checkpoint (RSL-RL style)")

    # =========================================================================
    # KEYBOARD CONTROLLER
    # =========================================================================

    keyboard_ctrl = QuadrupedKeyboardController()

    # Print controls
    print("\n" + "=" * 60)
    print("            KEYBOARD CONTROL ACTIVE")
    print("=" * 60)
    print("  Movement:")
    print("    W / UP / NUM8       - Forward")
    print("    S / DOWN / NUM2     - Backward")
    print("    A / LEFT / NUM4     - Strafe Left")
    print("    D / RIGHT / NUM6    - Strafe Right")
    print("  Rotation:")
    print("    Q / N / NUM7        - Turn Left")
    print("    E / M / NUM9        - Turn Right")
    print("  Control:")
    print("    R                   - Reset Robot")
    print("    ESC                 - Quit")
    print("=" * 60 + "\n")

    # =========================================================================
    # CONTROL LOOP
    # =========================================================================

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    step_count = 0

    while simulation_app.is_running() and not keyboard_ctrl.quit_requested:
        # Get velocity command from keyboard
        cmd = keyboard_ctrl.get_command(device)

        # Inject velocity command into environment
        try:
            if hasattr(unwrapped_env, "_commands"):
                unwrapped_env._commands[:] = cmd
        except Exception:
            pass

        # Get action from policy
        with torch.no_grad():
            # CRITICAL: Normalize observation if normalizer exists!
            if obs_normalizer is not None:
                obs_norm = obs_normalizer.normalize(obs)
            else:
                obs_norm = obs
            actions = actor_critic.act_inference(obs_norm)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Handle reset
        if keyboard_ctrl.reset_requested or dones.any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            keyboard_ctrl.reset_command()
            print("\n[RESET] Robot reset")

        # Print status every 5 steps
        if step_count % 5 == 0:
            keyboard_ctrl.print_status()

        step_count += 1

    # Cleanup
    env.close()
    print("\n[DONE] Keyboard control ended")


if __name__ == "__main__":
    main()
    simulation_app.close()