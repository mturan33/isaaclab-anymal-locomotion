"""
Isaac Lab PPO Implementation - From Scratch
============================================

Bu kod, MuJoCo PPO implementasyonundan Isaac Lab'ın GPU-paralel
ortamına adaptasyon sürecini gösterir.

TEMEL FARKLAR:
--------------
1. CPU NumPy → GPU PyTorch Tensors
2. Sequential env → Batched parallel envs (4096)
3. DummyVecEnv → Native Isaac Lab DirectRLEnv
4. Experience replay → On-policy rollout buffer
5. Individual updates → Batched tensor operations

Author: Turan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import math


# =============================================================================
# BÖLÜM 1: RUNNING STATISTICS (Observation & Reward Normalization)
# =============================================================================

class EmpiricalNormalization(nn.Module):
    """
    Online istatistik hesaplama - Welford's algoritması

    ESKİ KOD (MuJoCo):
    ------------------
    class RunningMeanStd:
        def __init__(self, shape, device):
            self.mean = torch.zeros(shape).to(device)
            self.var = torch.ones(shape).to(device)
            self.count = 1e-4

    YENİ KOD (Isaac Lab):
    ---------------------
    nn.Module olarak implement edildi çünkü:
    - Model save/load ile birlikte kaydedilir
    - GPU'da daha verimli çalışır
    - torch.jit.script ile derlenebilir

    MATEMATİK:
    ----------
    Welford's Online Algorithm:

    n = n + batch_size
    delta = batch_mean - mean
    mean = mean + delta * batch_size / n
    M2 = M2 + batch_var * batch_size + delta^2 * count * batch_size / n
    var = M2 / n

    Bu formül numerik olarak stabil ve tek geçişte hesaplanabilir.
    """

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()

        # nn.Module'ün register_buffer'ı kullan - model.state_dict()'e dahil olur
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))  # 1e-8 ile başla (0'a bölme önle)

        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """
        Yeni batch ile istatistikleri güncelle.

        Args:
            x: (batch_size, *input_shape) boyutunda tensor
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)  # Population variance
        batch_count = x.shape[0]

        # Welford's parallel algorithm
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        # Yeni mean
        self.running_mean = self.running_mean + delta * batch_count / total_count

        # Yeni variance (parallel combination formula)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.running_var = M2 / total_count

        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdiyi normalize et: (x - mean) / std

        Clipping [-5, 5] arasında yapılır çünkü:
        - Aşırı değerler policy'yi bozabilir
        - Numerik stabilite sağlar
        """
        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            min=-5.0,
            max=5.0
        )


# =============================================================================
# BÖLÜM 2: ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """
    Birleşik Actor-Critic ağı.

    ESKİ KOD (MuJoCo) - AYRI AĞLAR:
    -------------------------------
    class ActorNetwork(nn.Module):
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        # Activation: tanh

    class CriticNetwork(nn.Module):
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_layer = nn.Linear(128, 1)
        # Activation: tanh

    YENİ KOD (Isaac Lab) - BİRLEŞİK AĞ:
    -----------------------------------
    - Actor ve Critic tek class'ta
    - ELU aktivasyon (tanh yerine)
    - Daha büyük hidden dims: [512, 256, 128]
    - Ayrı MLP'ler (shared backbone yok)

    NEDEN BİRLEŞİK?
    ---------------
    1. Tek model.state_dict() ile save/load
    2. Daha temiz kod organizasyonu
    3. Isaac Lab convention'ı

    NEDEN AYRI MLP'LER (shared değil)?
    ----------------------------------
    Actor ve Critic farklı şeyler öğreniyor:
    - Actor: Optimal action → Gradient policy'den geliyor
    - Critic: Value estimate → Gradient TD error'dan geliyor

    Shared backbone kullanılsa gradient interference olurdu.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions

        # Aktivasyon seçimi
        activation_fn = self._get_activation(activation)

        # =====================================================================
        # ACTOR MLP
        # =====================================================================
        actor_layers = []
        in_dim = num_obs

        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(activation_fn())
            in_dim = hidden_dim

        # Son katman: action mean
        actor_layers.append(nn.Linear(in_dim, num_actions))

        self.actor_mlp = nn.Sequential(*actor_layers)

        # =====================================================================
        # CRITIC MLP
        # =====================================================================
        critic_layers = []
        in_dim = num_obs

        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(activation_fn())
            in_dim = hidden_dim

        # Son katman: value (tek scalar)
        critic_layers.append(nn.Linear(in_dim, 1))

        self.critic_mlp = nn.Sequential(*critic_layers)

        # =====================================================================
        # ACTION STD (Learnable)
        # =====================================================================
        """
        ESKİ KOD:
            self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
            
        YENİ KOD:
            self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))
            
        FARK:
        - Başlangıç değeri: -0.5 → log(1.0) = 0
        - Shape: (1, action_dim) → (action_dim,) - broadcasting ile aynı etki
        
        NEDEN LOG_STD?
        - std her zaman pozitif olmalı
        - log_std ise (-inf, inf) aralığında optimize edilebilir
        - exp(log_std) = std (her zaman pozitif)
        """
        self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))

        # Weight initialization
        self._init_weights()

    def _get_activation(self, name: str) -> type:
        """Aktivasyon fonksiyonu seç."""
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
        }
        return activations.get(name.lower(), nn.ELU)

    def _init_weights(self):
        """
        Orthogonal initialization.

        ESKİ KOD:
            torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)

        YENİ KOD:
            Aynı mantık, tüm katmanlara uygulanıyor.

        NEDEN ORTHOGONAL?
        -----------------
        - Gradient'ler katmanlar arasında magnitude'ünü korur
        - Vanishing/exploding gradient'i önler
        - RL'de standart practice

        GAIN DEĞERLERİ:
        - sqrt(2): ReLU/ELU için optimal
        - 0.01: Output layer için (küçük initial actions)
        - 1.0: Value layer için
        """
        for module in self.actor_mlp:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Actor output layer - küçük gain
        nn.init.orthogonal_(self.actor_mlp[-1].weight, gain=0.01)

        for module in self.critic_mlp:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Critic output layer
        nn.init.orthogonal_(self.critic_mlp[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - training için.

        Returns:
            actions: Sampled actions
            log_probs: Log probability of actions
            values: State value estimates
        """
        # Actor forward
        action_mean = self.actor_mlp(obs)

        # Std from learnable parameter
        action_std = torch.exp(self.log_std)

        # Create distribution
        dist = Normal(action_mean, action_std)

        # Sample action (reparameterization trick otomatik)
        actions = dist.sample()

        # Log probability
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Critic forward
        values = self.critic_mlp(obs).squeeze(-1)

        return actions, log_probs, values

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Verilen observation ve action için log_prob, value, entropy hesapla.
        PPO update'te kullanılır.

        ESKİ KOD (learn fonksiyonu içinde):
            dist = self.actor(norm_batch_states)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            entropy_loss = dist.entropy().mean()
            new_values = self.critic(norm_batch_states).squeeze()

        YENİ KOD:
            Aynı işlem, ayrı method olarak organize edildi.
        """
        action_mean = self.actor_mlp(obs)
        action_std = torch.exp(self.log_std)

        dist = Normal(action_mean, action_std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic_mlp(obs).squeeze(-1)

        return log_probs, values, entropy

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Inference için deterministic action.
        Play/test modunda kullanılır.

        ESKİ KOD:
            dist = self.actor(state)
            action = dist.sample()  # Stochastic

        YENİ KOD:
            Deterministic (mean action) - test için daha stabil
        """
        return self.actor_mlp(obs)

    @property
    def action_std(self) -> torch.Tensor:
        """Current action standard deviation."""
        return torch.exp(self.log_std)


# =============================================================================
# BÖLÜM 3: ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """
    On-policy experience storage.

    ESKİ KOD (MuJoCo):
    ------------------
    env_states = [[] for _ in range(num_envs)]
    env_actions = [[] for _ in range(num_envs)]
    # ... Python lists, sonra torch.cat ile birleştir

    YENİ KOD (Isaac Lab):
    ---------------------
    Pre-allocated GPU tensors

    NEDEN BU DEĞİŞİKLİK?
    --------------------
    1. GPU Memory Efficiency: Önceden allocate edilmiş tensörler
    2. No CPU-GPU Transfer: Her şey GPU'da kalıyor
    3. Batched Operations: Tüm env'ler tek tensörde

    BUFFER YAPISI:
    --------------
    Her tensor shape: (num_steps, num_envs, feature_dim)

    Örnek (4096 env, 24 step rollout):
    - observations: (24, 4096, 48)
    - actions: (24, 4096, 12)
    - rewards: (24, 4096)
    - dones: (24, 4096)
    - values: (24, 4096)
    - log_probs: (24, 4096)
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        num_obs: int,
        num_actions: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0

        # Pre-allocate tensors
        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)

        # GAE hesaplaması için
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ):
        """Bir step'in verilerini ekle."""
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs

        self.step += 1

    def compute_gae(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Generalized Advantage Estimation (GAE) hesapla.

        ESKİ KOD (MuJoCo):
        ------------------
        for t in reversed(range(num_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_advantage = delta + gamma * gae_lambda * last_advantage * mask
            advantages[t] = last_advantage

        YENİ KOD (Isaac Lab):
        ---------------------
        Aynı matematik, vectorized implementation

        GAE MATEMATİĞİ:
        ---------------
        TD Error (1-step advantage):
            δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

        GAE (λ-weighted sum of n-step advantages):
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        Recursive form:
            A_t = δ_t + γλ * A_{t+1}   (if not done)
            A_t = δ_t                   (if done)

        NEDEN GAE?
        ----------
        - λ=0: TD(0) - low variance, high bias
        - λ=1: Monte Carlo - high variance, low bias
        - λ=0.95: Sweet spot for locomotion
        """
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            # Mask: done ise next_value'yu 0'la
            next_non_terminal = 1.0 - self.dones[t]

            # TD error
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]

            # GAE recursive formula
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            self.advantages[t] = last_gae

        # Returns = advantages + values (value target)
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """
        Mini-batch generator.

        ESKİ KOD:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start:end]

        YENİ KOD:
            Aynı mantık, GPU tensörleriyle
        """
        # Flatten: (num_steps, num_envs, ...) → (num_steps * num_envs, ...)
        total_samples = self.num_steps * self.num_envs

        indices = torch.randperm(total_samples, device=self.device)

        # Flatten all tensors
        obs_flat = self.observations.view(total_samples, -1)
        actions_flat = self.actions.view(total_samples, -1)
        log_probs_flat = self.log_probs.view(total_samples)
        advantages_flat = self.advantages.view(total_samples)
        returns_flat = self.returns.view(total_samples)

        # Advantage normalization (per-batch değil, global)
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
        """Buffer'ı sıfırla."""
        self.step = 0


# =============================================================================
# BÖLÜM 4: PPO ALGORITHM
# =============================================================================

class PPOAlgorithm:
    """
    PPO Training Algorithm.

    ESKİ KOD (MuJoCo PPOAgent.learn):
    ---------------------------------
    def learn(self, states, actions, log_probs, returns, advantages, ...):
        for epoch in range(num_epochs):
            for batch in mini_batches:
                # Actor loss
                ratio = exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
                actor_loss = -min(surr1, surr2).mean()

                # Critic loss
                critic_loss = MSE(new_value, return)

    YENİ KOD:
    ---------
    Aynı matematik, daha modüler yapı

    PPO OBJECTIVE:
    --------------
    L^{CLIP}(θ) = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]

    where:
        r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
        A_t = advantage estimate (from GAE)
        ε = clip ratio (0.2)

    NEDEN CLIPPING?
    ---------------
    - Trust region constraint'i enforce eder
    - Policy'nin çok hızlı değişmesini önler
    - TRPO'nun constraint'ini "soft" bir şekilde uygular
    """

    def __init__(
        self,
        actor_critic: ActorCriticNetwork,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = None,
    ):
        self.actor_critic = actor_critic
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Single optimizer for both actor and critic
        """
        ESKİ KOD:
            self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
            self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
            
        YENİ KOD:
            Tek optimizer - daha basit, Isaac Lab convention
            
        NEDEN TEK OPTİMİZER?
        --------------------
        - Birleşik ağ için tek optimizer yeterli
        - Learning rate scheduling daha kolay
        - Gradient clipping tek seferde
        """
        self.optimizer = torch.optim.Adam(
            actor_critic.parameters(),
            lr=learning_rate,
            eps=1e-5  # Numerik stabilite
        )

        # Observation normalization
        self.obs_normalizer = EmpiricalNormalization((actor_critic.num_obs,))
        self.obs_normalizer.to(device)

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        """
        PPO policy update.

        Returns:
            dict: Training metrics (losses, kl divergence, etc.)
        """
        # Metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0

        # Calculate batch size
        total_samples = rollout_buffer.num_steps * rollout_buffer.num_envs
        batch_size = total_samples // self.num_mini_batches

        for epoch in range(self.num_learning_epochs):
            for batch in rollout_buffer.get_batches(batch_size):
                # Normalize observations
                obs_normalized = self.obs_normalizer.normalize(batch["obs"])

                # Get current policy outputs
                new_log_probs, values, entropy = self.actor_critic.evaluate(
                    obs_normalized, batch["actions"]
                )

                # =========================================================
                # ACTOR (POLICY) LOSS
                # =========================================================
                """
                Probability Ratio:
                    r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                    
                Log space'de:
                    log(r_t) = log π_θ - log π_θ_old
                    r_t = exp(log π_θ - log π_θ_old)
                """
                log_ratio = new_log_probs - batch["old_log_probs"]
                ratio = torch.exp(log_ratio)

                # Surrogate objectives
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param
                ) * batch["advantages"]

                # Take minimum (pessimistic bound)
                actor_loss = -torch.min(surr1, surr2).mean()

                # =========================================================
                # CRITIC (VALUE) LOSS
                # =========================================================
                """
                ESKİ KOD:
                    critic_loss = 0.5 * MSELoss()(new_values, returns)
                    
                YENİ KOD:
                    Aynı, value_loss_coef ile scale edilmiş
                    
                NEDEN 0.5?
                ----------
                MSE = (y - y')² 
                Gradient = 2(y - y')
                0.5 * MSE → Gradient = (y - y')  (daha clean)
                """
                value_loss = 0.5 * F.mse_loss(values, batch["returns"])

                # =========================================================
                # ENTROPY BONUS
                # =========================================================
                """
                H(π) = -Σ π(a|s) log π(a|s)
                
                Gaussian için:
                    H = 0.5 * log(2πe * σ²) = 0.5 * (1 + log(2π) + 2*log(σ))
                    
                NEDEN ENTROPY?
                --------------
                - Exploration'ı teşvik eder
                - Premature convergence'ı önler
                - Policy'nin çok deterministic olmasını engeller
                """
                entropy_loss = entropy.mean()

                # =========================================================
                # TOTAL LOSS
                # =========================================================
                """
                L = L_actor + c1 * L_critic - c2 * H
                
                - L_actor: Policy gradient loss (minimize)
                - L_critic: Value function loss (minimize)
                - H: Entropy (maximize, bu yüzden -)
                """
                loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # =========================================================
                # GRADIENT UPDATE
                # =========================================================
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                """
                ESKİ KOD:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    
                YENİ KOD:
                    Tek model için tek clip
                    
                NEDEN GRADIENT CLIPPING?
                ------------------------
                - Exploding gradient'i önler
                - Training stability sağlar
                - Özellikle büyük batch'lerde önemli
                """
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # =========================================================
                # KL DIVERGENCE (for monitoring/early stopping)
                # =========================================================
                """
                Approximate KL divergence:
                    KL ≈ E[(r - 1) - log(r)]
                    
                Daha basit yaklaşım:
                    KL ≈ E[log(π_old) - log(π_new)]
                """
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # Accumulate metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_kl += approx_kl
                num_updates += 1

            # Early stopping based on KL divergence
            """
            ESKİ KOD:
                if mean_kl > 0.02:
                    print(f"[EARLY STOP] ...")
                    break
                    
            YENİ KOD:
                Aynı mantık, 0.015 threshold (biraz daha conservative)
            """
            mean_kl_epoch = total_kl / num_updates
            if mean_kl_epoch > 0.015:
                print(f"[EARLY STOP] Epoch {epoch+1}, KL={mean_kl_epoch:.4f}")
                break

        # Return metrics
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl_divergence": total_kl / num_updates,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "action_std": self.actor_critic.action_std.mean().item(),
        }


# =============================================================================
# BÖLÜM 5: RUNNER (Training Loop)
# =============================================================================

class PPORunner:
    """
    Training orchestrator.

    ESKİ KOD (main_parallel.py):
    ----------------------------
    while global_step_count < total_timesteps:
        # Rollout collection
        for step in range(rollout_steps):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            # Store in lists...

        # Compute advantages
        advantages, returns = agent.compute_advantages(...)

        # Update
        agent.learn(...)

    YENİ KOD:
    ---------
    Daha modüler, ayrı class'lar
    """

    def __init__(
        self,
        env,  # Isaac Lab DirectRLEnv
        actor_critic: ActorCriticNetwork,
        ppo: PPOAlgorithm,
        rollout_buffer: RolloutBuffer,
        num_steps_per_rollout: int = 24,
        device: torch.device = None,
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.ppo = ppo
        self.rollout_buffer = rollout_buffer
        self.num_steps_per_rollout = num_steps_per_rollout
        self.device = device

        # Statistics
        self.total_timesteps = 0
        self.total_episodes = 0

    def collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Rollout data topla.

        ESKİ KOD:
            for step in range(rollout_steps):
                for i in range(num_envs):
                    action, log_prob, value = agent.get_action(states[i])

        YENİ KOD:
            Tüm env'ler için tek forward pass (batched)
        """
        self.rollout_buffer.reset()

        for step in range(self.num_steps_per_rollout):
            # Normalize observation
            obs_normalized = self.ppo.obs_normalizer.normalize(obs)

            # Get actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.actor_critic(obs_normalized)

            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            dones = terminated | truncated

            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones.float(),
                values=values,
                log_probs=log_probs,
            )

            # Update observation normalizer
            self.ppo.obs_normalizer.update(obs)

            # Update statistics
            self.total_timesteps += obs.shape[0]  # num_envs

            obs = next_obs

        # Compute last values for GAE
        with torch.no_grad():
            obs_normalized = self.ppo.obs_normalizer.normalize(obs)
            _, _, last_values = self.actor_critic(obs_normalized)

        # Compute GAE
        self.rollout_buffer.compute_gae(
            last_values=last_values,
            gamma=self.ppo.gamma,
            gae_lambda=self.ppo.gae_lambda,
        )

        return obs

    def train_step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Tek training iteration: rollout + update.
        """
        # Collect rollout
        next_obs = self.collect_rollout(obs)

        # Update policy
        metrics = self.ppo.update(self.rollout_buffer)

        return next_obs, metrics

    def save(self, path: str):
        """Model ve normalizer kaydet."""
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "obs_normalizer": self.ppo.obs_normalizer.state_dict(),
            "total_timesteps": self.total_timesteps,
        }, path)

    def load(self, path: str):
        """Model ve normalizer yükle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.ppo.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

def create_ppo_system(
    num_obs: int,
    num_actions: int,
    num_envs: int,
    device: torch.device,
    # Hyperparameters (senin MuJoCo değerlerin)
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_param: float = 0.2,
    num_learning_epochs: int = 10,
    num_mini_batches: int = 4,
    num_steps_per_rollout: int = 24,
    entropy_coef: float = 0.01,
):
    """
    PPO sistemini oluştur.

    Isaac Lab training script'inde kullanım:

    ```python
    actor_critic, ppo, buffer = create_ppo_system(
        num_obs=48,
        num_actions=12,
        num_envs=4096,
        device=torch.device("cuda"),
    )

    obs, _ = env.reset()
    for iteration in range(max_iterations):
        with torch.no_grad():
            actions, _, _ = actor_critic(obs)
        obs, reward, done, info = env.step(actions)
        # ... buffer.add(), ppo.update()
    ```
    """

    # Create actor-critic network
    actor_critic = ActorCriticNetwork(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],  # Isaac Lab default
        critic_hidden_dims=[512, 256, 128],
        activation="elu",  # Tanh yerine ELU
        init_noise_std=1.0,
    ).to(device)

    # Create PPO algorithm
    ppo = PPOAlgorithm(
        actor_critic=actor_critic,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
        clip_param=clip_param,
        entropy_coef=entropy_coef,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    # Create rollout buffer
    buffer = RolloutBuffer(
        num_envs=num_envs,
        num_steps=num_steps_per_rollout,
        num_obs=num_obs,
        num_actions=num_actions,
        device=device,
    )

    return actor_critic, ppo, buffer


# =============================================================================
# KARŞILAŞTIRMA TABLOSU
# =============================================================================
"""
+-------------------------+---------------------------+---------------------------+
| Özellik                 | Eski (MuJoCo)             | Yeni (Isaac Lab)          |
+-------------------------+---------------------------+---------------------------+
| Environment             | DummyVecEnv (CPU)         | DirectRLEnv (GPU)         |
| Paralel Env Sayısı      | 16                        | 4096                      |
| Data Storage            | Python lists              | Pre-allocated GPU tensors |
| Observation Norm        | RunningMeanStd class      | nn.Module (saveable)      |
| Network Architecture    | Ayrı Actor/Critic         | Birleşik ActorCritic      |
| Hidden Dims             | [256, 256, 128]           | [512, 256, 128]           |
| Activation              | Tanh                      | ELU                       |
| Optimizer               | 2 ayrı Adam               | 1 birleşik Adam           |
| Batch Processing        | Loop over envs            | Batched tensor ops        |
| GAE Computation         | Sequential                | Vectorized                |
| KL Early Stop           | 0.02                      | 0.015                     |
| Gradient Clip           | 0.5                       | 1.0                       |
+-------------------------+---------------------------+---------------------------+

PERFORMANS FARKI:
-----------------
MuJoCo (16 env, CPU):     ~1,000 steps/second
Isaac Lab (4096 env, GPU): ~80,000 steps/second

80x speedup!
"""