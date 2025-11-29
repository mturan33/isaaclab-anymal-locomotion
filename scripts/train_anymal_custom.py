"""
Custom PPO Training for Isaac Lab - RSL-RL Style Logging
=========================================================

From-scratch PPO implementation with full RSL-RL compatible logging.

Usage:
    ./isaaclab.bat -p scripts/train_anymal_custom.py --task Isaac-MyAnymal-Flat-v0 --num_envs 4096 --headless
"""

from __future__ import annotations

import argparse
import os
import time
import math
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Custom PPO Training")
parser.add_argument("--task", type=str, default="Isaac-MyAnymal-Flat-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--experiment_name", type=str, default="custom_ppo_v2", help="Experiment name")
parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval")
parser.add_argument("--log_interval", type=int, default=1, help="Log interval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Enable TF32 for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def get_obs_tensor(obs):
    """Extract observation tensor from dict."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def format_time(seconds):
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =============================================================================
# EMPIRICAL NORMALIZATION (Welford's Algorithm)
# =============================================================================

class EmpiricalNormalization(nn.Module):
    """Online observation normalization."""

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.running_var = M2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            min=-5.0, max=5.0
        )


# =============================================================================
# ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic with ELU activation."""

    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128], init_noise_std=1.0):
        super().__init__()

        # Actor
        actor_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable std
        self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))

        self._init_weights()

    def _init_weights(self):
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        return actions, log_probs, values

    def evaluate(self, obs, actions):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        return log_probs, values, entropy

    def act_inference(self, obs):
        return self.actor(obs)


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """On-policy rollout storage."""

    def __init__(self, num_envs, num_steps, num_obs, num_actions, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0

        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.step += 1

    def compute_gae(self, last_values, gamma, gae_lambda):
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)

        obs_flat = self.observations.view(total_samples, -1)
        actions_flat = self.actions.view(total_samples, -1)
        log_probs_flat = self.log_probs.view(total_samples)
        advantages_flat = self.advantages.view(total_samples)
        returns_flat = self.returns.view(total_samples)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "obs": obs_flat[batch_idx],
                "actions": actions_flat[batch_idx],
                "old_log_probs": log_probs_flat[batch_idx],
                "advantages": advantages_flat[batch_idx],
                "returns": returns_flat[batch_idx],
            }

    def reset(self):
        self.step = 0


# =============================================================================
# TRAINING
# =============================================================================

def train():
    # =========================================================================
    # HYPERPARAMETERS
    # =========================================================================
    learning_rate = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_rollout = 24
    value_loss_coef = 1.0
    max_grad_norm = 1.0

    init_noise_std = 1.0
    entropy_coef = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args_cli.seed)

    # =========================================================================
    # LOGGING SETUP
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = os.path.join("logs", "rsl_rl", args_cli.experiment_name)
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    writer = SummaryWriter(log_dir)

    print(f"[INFO] Logging experiment in directory: {os.path.abspath(log_root)}")
    print(f"Exact experiment name requested from command line: {timestamp}")

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Get unwrapped env for device and extras
    unwrapped_env = env.unwrapped

    num_obs = env.observation_space.shape[1]
    num_actions = env.action_space.shape[1]
    num_envs = args_cli.num_envs

    print(f"[INFO] Observation dim: {num_obs}")
    print(f"[INFO] Action dim: {num_actions}")
    print(f"[INFO] Num envs: {num_envs}")

    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    actor_critic = ActorCriticNetwork(
        num_obs=num_obs,
        num_actions=num_actions,
        hidden_dims=[128, 128, 128],
        init_noise_std=1.0,
    ).to(device)

    obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate, eps=1e-5)
    print(f"[INFO] Optimizer params: {sum(p.numel() for p in actor_critic.parameters())} (log_std included - learnable)")

    # LR scheduler
    def lr_lambda(iteration):
        return max(0.1, 1.0 - iteration / args_cli.max_iterations)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Rollout buffer
    buffer = RolloutBuffer(num_envs, num_steps_per_rollout, num_obs, num_actions, device)

    # Load checkpoint
    start_iteration = 0
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        start_iteration = checkpoint.get("iteration", 0)
        print(f"[INFO] Resuming from iteration {start_iteration}")

    # =========================================================================
    # TRACKING VARIABLES
    # =========================================================================
    total_timesteps = start_iteration * num_steps_per_rollout * num_envs
    best_reward = -float("inf")

    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device)

    # Reward components tracking
    reward_components = {}
    termination_components = {}

    # Reset environment
    obs_dict, info = env.reset()
    obs = get_obs_tensor(obs_dict)

    training_start_time = time.time()

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start_time = time.time()
        collection_start = time.time()

        # =====================================================================
        # COLLECT ROLLOUT
        # =====================================================================
        buffer.reset()
        rollout_rewards = []

        for step in range(num_steps_per_rollout):
            obs_norm = obs_normalizer.normalize(obs)

            with torch.no_grad():
                actions, log_probs, values = actor_critic(obs_norm)

            next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
            next_obs = get_obs_tensor(next_obs_dict)
            dones = (terminated | truncated).float()

            buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs_normalizer.update(obs)

            # Track episode stats
            current_episode_rewards += rewards
            current_episode_lengths += 1

            # Handle episode ends
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                for idx in done_indices:
                    episode_rewards.append(current_episode_rewards[idx].item())
                    episode_lengths.append(current_episode_lengths[idx].item())
                current_episode_rewards[done_indices] = 0
                current_episode_lengths[done_indices] = 0

            # Extract reward/termination components from extras
            if "log" in info:
                log_info = info["log"]
                for key, value in log_info.items():
                    if "Episode_Reward" in key or "reward" in key.lower():
                        if key not in reward_components:
                            reward_components[key] = deque(maxlen=100)
                        if isinstance(value, torch.Tensor):
                            reward_components[key].append(value.mean().item())
                        else:
                            reward_components[key].append(value)
                    elif "Episode_Termination" in key or "termination" in key.lower():
                        if key not in termination_components:
                            termination_components[key] = deque(maxlen=100)
                        if isinstance(value, torch.Tensor):
                            termination_components[key].append(value.sum().item())
                        else:
                            termination_components[key].append(value)

            rollout_rewards.append(rewards.mean().item())
            total_timesteps += num_envs
            obs = next_obs

        collection_time = time.time() - collection_start

        # =====================================================================
        # COMPUTE GAE
        # =====================================================================
        with torch.no_grad():
            obs_norm = obs_normalizer.normalize(obs)
            _, _, last_values = actor_critic(obs_norm)
        buffer.compute_gae(last_values, gamma, gae_lambda)

        # =====================================================================
        # PPO UPDATE
        # =====================================================================
        learning_start = time.time()

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = (num_steps_per_rollout * num_envs) // num_mini_batches

        for epoch in range(num_learning_epochs):
            for batch in buffer.get_batches(batch_size):
                obs_norm = obs_normalizer.normalize(batch["obs"])
                new_log_probs, values, entropy = actor_critic.evaluate(obs_norm, batch["actions"])

                # Actor loss
                log_ratio = new_log_probs - batch["old_log_probs"]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = 0.5 * F.mse_loss(values, batch["returns"])

                # Entropy
                entropy_loss = entropy.mean()

                # Total loss
                loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
                optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        scheduler.step()

        learning_time = time.time() - learning_start

        # =====================================================================
        # LOGGING
        # =====================================================================
        iter_time = time.time() - iter_start_time
        elapsed_time = time.time() - training_start_time
        remaining_iters = args_cli.max_iterations - iteration - 1
        eta = remaining_iters * iter_time

        mean_reward = sum(rollout_rewards) / len(rollout_rewards)
        mean_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_episode_length = np.mean(episode_lengths) if episode_lengths else 0

        steps_per_sec = (num_steps_per_rollout * num_envs) / iter_time
        mean_std = actor_critic.log_std.exp().mean().item()
        mean_actor_loss = total_actor_loss / num_updates
        mean_critic_loss = total_critic_loss / num_updates
        mean_entropy = total_entropy / num_updates

        writer.add_scalar("Loss/surrogate", mean_actor_loss, iteration)
        writer.add_scalar("Loss/value_function", mean_critic_loss, iteration)
        writer.add_scalar("Loss/entropy", mean_entropy, iteration)
        writer.add_scalar("Loss/learning_rate", scheduler.get_last_lr()[0], iteration)
        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Train/mean_episode_reward", mean_episode_reward, iteration)
        writer.add_scalar("Train/mean_episode_length", mean_episode_length, iteration)
        writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        writer.add_scalar("Perf/total_fps", steps_per_sec, iteration)
        writer.add_scalar("Perf/collection time", collection_time, iteration)
        writer.add_scalar("Perf/learning_time", learning_time, iteration)

        # Log reward components
        for key, values_deque in reward_components.items():
            if values_deque:
                writer.add_scalar(key, np.mean(values_deque), iteration)

        # Log termination components
        for key, values_deque in termination_components.items():
            if values_deque:
                writer.add_scalar(key, np.mean(values_deque), iteration)

        # Console output (RSL-RL style)
        if iteration % args_cli.log_interval == 0:
            print("#" * 80)
            print(f"{'Learning iteration ' + str(iteration) + '/' + str(args_cli.max_iterations):^80}")
            print(f"{'Computation: ' + str(int(steps_per_sec)) + ' steps/s (collection: ' + f'{collection_time:.3f}s, learning ' + f'{learning_time:.3f}s)':^80}")
            print(f"{'Mean action noise std:':>35} {mean_std:.2f}")
            print(f"{'Mean value_function loss:':>35} {mean_critic_loss:.4f}")
            print(f"{'Mean surrogate loss:':>35} {mean_actor_loss:.4f}")
            print(f"{'Mean entropy loss:':>35} {mean_entropy:.4f}")
            print(f"{'Mean reward:':>35} {mean_reward:.2f}")
            print(f"{'Mean episode length:':>35} {mean_episode_length:.2f}")

            # Print reward components
            for key, values_deque in sorted(reward_components.items()):
                if values_deque:
                    print(f"{key:>40}: {np.mean(values_deque):.4f}")

            # Print termination components
            for key, values_deque in sorted(termination_components.items()):
                if values_deque:
                    print(f"{key:>40}: {np.mean(values_deque):.4f}")

            print("-" * 80)
            print(f"{'Total timesteps:':>35} {total_timesteps}")
            print(f"{'Iteration time:':>35} {iter_time:.2f}s")
            print(f"{'Time elapsed:':>35} {format_time(elapsed_time)}")
            print(f"{'ETA:':>35} {format_time(eta)}")
            print("#" * 80)
            print()

        # =====================================================================
        # SAVE CHECKPOINTS
        # =====================================================================
        checkpoint_data = {
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
            "iteration": iteration,
        }

        # Best model
        if mean_episode_reward > best_reward:
            best_reward = mean_episode_reward
            torch.save(checkpoint_data, os.path.join(log_dir, "model_best.pt"))

        # Periodic checkpoint
        if iteration % args_cli.save_interval == 0 and iteration > 0:
            torch.save(checkpoint_data, os.path.join(log_dir, f"model_{iteration}.pt"))
            print(f"[CHECKPOINT] Saved model_{iteration}.pt")

    # =========================================================================
    # FINAL SAVE
    # =========================================================================
    torch.save(checkpoint_data, os.path.join(log_dir, f"model_{args_cli.max_iterations}.pt"))

    total_time = time.time() - training_start_time
    print("\n" + "=" * 80)
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"{'Total time: ' + format_time(total_time):^80}")
    print(f"{'Best reward: ' + f'{best_reward:.2f}':^80}")
    print(f"{'Log dir: ' + log_dir:^80}")
    print("=" * 80)

    writer.close()
    env.close()


if __name__ == "__main__":
    train()
    simulation_app.close()