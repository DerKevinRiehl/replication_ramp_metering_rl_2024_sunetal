"""
DDPG agent for the Combined MPC-DRL Framework (Sun et al., 2024).

This agent is **not** a Controller subclass.  The hierarchical triple-nested
training loop in ``train_mpc_drl.py`` orchestrates the MPC and DRL modules
externally, calling this agent's methods at the DRL frequency (T_d = 60 s).

Key differences from the standalone DDPGController
---------------------------------------------------
* Actor outputs in [-1, 1] (tanh), then scaled by  w_u · ΔU.
* The MPC baseline u_b is injected externally; the saturation function
  computes  u_c = clip(u_b + u_rl, U_min, U_max).
* State augmentation fills ū_s with the *normalised MPC output* instead
  of zeros.
* Exploration noise has std = 0.2 and exponential decay rate 2e-5 per
  DRL step (paper §IV-B.5).
* The ``estimation_step`` parameter toggles 1-step vs 10-step TD.

Architecture
------------
Actor :  30 → 256 → 256 → 3   (tanh scaled to [-w_u·ΔU, +w_u·ΔU])
Critic:  state(30 → 256) ⊕ action(3 → 128) → 256 → 128 → 1
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg_networks import Critic
from controllers.replay_buffer import NStepReplayBuffer
from config import (
    STATE_DIM, ACTION_DIM,
    GAMMA, TAU, LR_ACTOR, LR_CRITIC,
    BUFFER_SIZE, BATCH_SIZE,
    N_DEMAND_LOOKAHEAD,
    RHO_MAX, V_MAX, W_MAX_MAIN, W_MAX_RAMP,
    D_MAIN_MAX, D_RAMP_MAX, U_VSL_MAX, U_RAMP_MAX,
    DDPG_STEP,
)


# ======================================================================
# Actor with tanh output for MPC-DRL framework
# ======================================================================

class ActorMPCDRL(nn.Module):
    """
    Deterministic policy  π_θ : S → A   for the MPC-DRL combined framework.

    Architecture (same hidden layers as standalone Actor):
        Input(30) → 256 → 256 → Output(3)
        Hidden activations : ReLU
        Output activation  : tanh  →  range [-1, +1]

    The raw [-1, 1] output is then scaled externally by  w_u · ΔU.
    """

    def __init__(self, state_dim: int = 30, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),                   # output in [-1, +1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw action in [-1, +1]."""
        return self.net(state)


# ======================================================================
# DDPG MPC Agent
# ======================================================================

class DDPGMPCAgent:
    """
    DDPG agent for the hierarchical MPC-DRL framework.

    Parameters
    ----------
    demand : np.ndarray, shape (n_sim_steps, 2)
        Base (noise-free) demand profile for state augmentation.
    estimation_step : int
        Number of TD steps (1 or 10).  When ``estimation_step=10``, the
        replay buffer computes 10-step discounted returns, and the critic
        target uses γ^10.
    device : str
        PyTorch device ("cpu" or "cuda").
    """

    # --- Paper §IV-B.5 MPC-DRL specific constants ---
    W_U = 0.4                           # scaling parameter
    DELTA_U_VSL = 0.8 * U_VSL_MAX       # 0.8 × 102 = 81.6 km/h
    DELTA_U_RAMP = U_RAMP_MAX           # 1.0
    NOISE_STD_INIT = 0.2                # exploration noise std
    NOISE_DECAY = 2e-5                  # exponential decay rate per DRL step

    # Action bounds for the *overall* combined control
    U_MIN = np.array([20.0, 20.0, 0.0], dtype=np.float32)
    U_MAX = np.array([U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX], dtype=np.float32)

    # DRL action bounds  (±w_u · ΔU)
    # VSL: ±0.4 × 81.6 = ±32.64    Ramp: ±0.4 × 1.0 = ±0.4
    ACTION_SCALE = np.array([
        W_U * 0.8 * U_VSL_MAX,     # 32.64
        W_U * 0.8 * U_VSL_MAX,     # 32.64
        W_U * U_RAMP_MAX,          # 0.4
    ], dtype=np.float32)

    def __init__(
        self,
        demand: np.ndarray,
        estimation_step: int = 10,
        device: str = "cpu",
    ):
        self.demand = np.asarray(demand)
        self.estimation_step = estimation_step
        self.device = device

        # ---- Networks ----
        self.actor = ActorMPCDRL(STATE_DIM, ACTION_DIM).to(device)
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Freeze target networks
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Optimisers ----
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=LR_ACTOR
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=LR_CRITIC
        )

        # ---- Replay buffer ----
        self.buffer = NStepReplayBuffer(
            capacity=BUFFER_SIZE,
            n_step=estimation_step,
            gamma=GAMMA,
        )

        # ---- Internal state ----
        self._prev_action = np.array(
            [U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX], dtype=np.float32
        )                                        # u_c(k_d - 1)
        self._noise_std = self.NOISE_STD_INIT
        self._total_drl_steps = 0                # global step counter for decay

    # ------------------------------------------------------------------
    # Reset (called at start of each episode)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode internal state."""
        self._prev_action = np.array(
            [U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # State construction  (paper eq. 7)
    # ------------------------------------------------------------------
    def build_state(
        self,
        obs: np.ndarray,
        sim_step: int,
        mpc_action: np.ndarray,
    ) -> np.ndarray:
        """
        Build the 30-dim normalised DRL state.

        Components
        ----------
        x̄     : normalised  [ρ(6), v(6), w(2)]                 → 14
        ū_s   : normalised MPC output  [vsl1/V, vsl2/V, r]     → 3
        d̄     : normalised demand forecast  (5 steps × 2)      → 10
        ū_c   : normalised previous overall control input       → 3
                                                           Total: 30

        Parameters
        ----------
        obs : np.ndarray, shape (14,)
            Raw env observation [rho(6), v(6), w(2)].
        sim_step : int
            Current simulation step index (k_d * m_2).
        mpc_action : np.ndarray, shape (3,)
            MPC baseline output u_b = [vsl1, vsl2, r].
        """
        # 1. Normalised freeway states  (14 dims)
        rho = obs[:6] / RHO_MAX
        v = obs[6:12] / V_MAX
        w = np.array([
            obs[12] / W_MAX_MAIN,
            obs[13] / W_MAX_RAMP,
        ])
        x_bar = np.concatenate([rho, v, w])                      # (14,)

        # 2. Normalised MPC output  (3 dims)
        u_s_bar = np.array([
            mpc_action[0] / U_VSL_MAX,
            mpc_action[1] / U_VSL_MAX,
            mpc_action[2] / U_RAMP_MAX,
        ])                                                        # (3,)

        # 3. Demand forecast  (10 dims = 5 steps × 2)
        d_forecast = self._get_demand_forecast(sim_step)          # (10,)

        # 4. Previous overall control input  (3 dims, normalised)
        u_c_bar = np.array([
            self._prev_action[0] / U_VSL_MAX,
            self._prev_action[1] / U_VSL_MAX,
            self._prev_action[2] / U_RAMP_MAX,
        ])                                                        # (3,)

        return np.concatenate([x_bar, u_s_bar, d_forecast, u_c_bar]).astype(
            np.float32
        )

    def _get_demand_forecast(self, step_idx: int) -> np.ndarray:
        """
        Get normalised demand forecast: N_DEMAND_LOOKAHEAD steps ahead,
        sampled at the DRL frequency (every DDPG_STEP simulation steps).
        """
        n_sim = self.demand.shape[0]
        forecast = np.zeros(N_DEMAND_LOOKAHEAD * 2)

        for i in range(N_DEMAND_LOOKAHEAD):
            future_idx = step_idx + i * int(DDPG_STEP)
            if future_idx < n_sim:
                forecast[2 * i] = self.demand[future_idx, 0] / D_MAIN_MAX
                forecast[2 * i + 1] = self.demand[future_idx, 1] / D_RAMP_MAX
            else:
                # Pad with last known demand
                forecast[2 * i] = self.demand[-1, 0] / D_MAIN_MAX
                forecast[2 * i + 1] = self.demand[-1, 1] / D_RAMP_MAX

        return forecast

    # ------------------------------------------------------------------
    # Action selection  (paper §III-B)
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select DRL action  u_rl(k_d).

        Returns the *raw DRL adjustment* (before combining with MPC).
        Shape (3,), values in [-ACTION_SCALE, +ACTION_SCALE].
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            raw = self.actor(state_t).cpu().numpy().reshape(-1)   # [-1, +1]
        self.actor.train()

        # Scale to the DRL action bounds:  u_rl = raw * (w_u · ΔU)
        u_rl = raw * self.ACTION_SCALE

        # Add exploration noise  w_n ~ N(0, σ)
        if add_noise:
            noise = np.random.normal(0, self._noise_std, size=u_rl.shape)
            u_rl = u_rl + noise
            # Clip to action bounds
            u_rl = np.clip(u_rl, -self.ACTION_SCALE, self.ACTION_SCALE)

            # Decay noise (exponential decay per DRL step)
            self._total_drl_steps += 1
            self._noise_std = self.NOISE_STD_INIT * np.exp(
                -self.NOISE_DECAY * self._total_drl_steps
            )

        return u_rl.astype(np.float32)

    # ------------------------------------------------------------------
    # Saturation function  (paper eq. 5)
    # ------------------------------------------------------------------
    @staticmethod
    def saturate(u_b: np.ndarray, u_rl: np.ndarray) -> np.ndarray:
        """
        Combine MPC baseline and DRL adjustment with saturation.

            u_c = clip(u_b + u_rl,  U_min, U_max)
        """
        return np.clip(
            u_b + u_rl,
            DDPGMPCAgent.U_MIN,
            DDPGMPCAgent.U_MAX,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Replay buffer interface
    # ------------------------------------------------------------------
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update_prev_action(self, u_c: np.ndarray) -> None:
        """Update the previous *combined* action (for state augmentation)."""
        self._prev_action = u_c.copy()

    # ------------------------------------------------------------------
    # Training  (DDPG update)
    # ------------------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        """
        One DDPG update:  critic → actor → soft target update.

        Returns dict with ``critic_loss`` and ``actor_loss`` for logging.
        """
        if self.buffer.size < BATCH_SIZE:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = self.buffer.sample(BATCH_SIZE, device=self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            # Target actor outputs raw [-1, 1]; scale to DRL action bounds
            next_raw = self.actor_target(next_states)             # [-1, +1]
            action_scale_t = torch.tensor(
                self.ACTION_SCALE, dtype=torch.float32, device=self.device
            )
            next_actions = next_raw * action_scale_t

            q_target = self.critic_target(next_states, next_actions)
            gamma_n = GAMMA ** self.estimation_step
            y = rewards + gamma_n * (1.0 - dones) * q_target

        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- Actor update (policy gradient) ---
        raw_actions = self.actor(states)                          # [-1, +1]
        action_scale_t = torch.tensor(
            self.ACTION_SCALE, dtype=torch.float32, device=self.device
        )
        scaled_actions = raw_actions * action_scale_t
        actor_loss = -self.critic(states, scaled_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- Soft target updates ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    @staticmethod
    def _soft_update(
        source: torch.nn.Module, target: torch.nn.Module
    ) -> None:
        """θ' ← τ θ + (1-τ) θ'"""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(TAU * sp.data + (1.0 - TAU) * tp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save actor, critic, optimiser state dicts and noise state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "noise_std": self._noise_std,
                "total_drl_steps": self._total_drl_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load actor, critic, optimiser state dicts and noise state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "noise_std" in ckpt:
            self._noise_std = ckpt["noise_std"]
        if "total_drl_steps" in ckpt:
            self._total_drl_steps = ckpt["total_drl_steps"]
