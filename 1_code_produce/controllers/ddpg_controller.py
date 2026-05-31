"""
DDPG Controller – standalone DRL agent for freeway traffic control.

Implements the Controller ABC.  Supports both training (with exploration
noise and train_step) and inference (deterministic policy).

State augmentation (paper eq. 7):
    x_rl = [ x̄(14),  ū_s(3)=0,  d̄(10),  ū_c(3) ]  →  30-dim
    All components are normalised to [0, 1].
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from controllers.base_controller import Controller
from controllers.ddpg_networks import Actor, Critic
from controllers.replay_buffer import NStepReplayBuffer
from config import (
    DDPG_STEP, STEPS_PER_EP,
    STATE_DIM, ACTION_DIM,
    GAMMA, TAU, LR_ACTOR, LR_CRITIC,
    EXPLORATION_NOISE, BUFFER_SIZE, BATCH_SIZE,
    N_DEMAND_LOOKAHEAD,
    RHO_MAX, V_MAX, W_MAX_MAIN, W_MAX_RAMP,
    D_MAIN_MAX, D_RAMP_MAX, U_VSL_MAX, U_RAMP_MAX,
)


class DDPGController(Controller):
    """
    DDPG-based standalone DRL controller.

    Parameters
    ----------
    demand : np.ndarray, shape (n_sim_steps, 2)
        Base (noise-free) demand profile for state augmentation.
    n_step : int
        TD steps (1 or 10).
    training : bool
        If True, adds exploration noise and allows train_step().
    device : str
        PyTorch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        demand: np.ndarray,
        n_step: int = 1,
        training: bool = True,
        device: str = "cpu",
        model_dir: str | None = None,
        model_tag: str | None = None,
        model_type: str = "best",
        **kwargs,                    # accept (and ignore) extra kwargs from run_experiments
    ):
        self.demand = np.asarray(demand)
        self.n_step = n_step
        self.training = training
        self.device = device
        self._model_dir = model_dir
        self._model_tag = model_tag

        # Networks
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(device)
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Freeze target network gradients
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=LR_ACTOR
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=LR_CRITIC
        )

        # Replay buffer
        self.buffer = NStepReplayBuffer(
            capacity=BUFFER_SIZE, n_step=n_step, gamma=GAMMA
        )

        # Internal state
        self._prev_action = np.array([U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX])
        self._cached_action = self._prev_action.copy()
        self._current_step = 0              # simulation step index
        self._noise_std = EXPLORATION_NOISE
        self._total_drl_steps = 0           # global counter for noise decay
        self._noise_decay = 2e-5            # exponential decay rate per DRL step

        # Action normalisation scales (for critic input)
        self._action_norm = np.array([U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX],
                                     dtype=np.float32)

        # Auto-load model if model_dir and model_tag provided (evaluation mode)
        if model_dir is not None and model_tag is not None:
            import os
            load_path = os.path.join(model_dir, f"{model_tag}_{model_type}.pt")
            if os.path.isfile(load_path):
                self.load(load_path)
                print(f"  [DDPG] Loaded model from {load_path}")
            else:
                print(f"  [DDPG] WARNING: No model found at {load_path}")

    # ------------------------------------------------------------------
    # Controller ABC
    # ------------------------------------------------------------------
    @property
    def update_interval(self) -> int:
        return int(DDPG_STEP)

    def reset(self) -> None:
        self._prev_action = np.array([U_VSL_MAX, U_VSL_MAX, U_RAMP_MAX])
        self._cached_action = self._prev_action.copy()
        self._current_step = 0

    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Return the DRL action for this simulation step.

        Re-computes at every ``DDPG_STEP``-th step; otherwise returns cached.
        """
        self._current_step = step_idx

        if step_idx % self.update_interval != 0:
            return self._cached_action.copy()

        state = self.build_state(obs, step_idx)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().reshape(-1)
        self.actor.train()

        # Exploration noise during training
        if self.training:
            noise = np.random.normal(0, self._noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, [20.0, 20.0, 0.0], [102.0, 102.0, 1.0])

            # Decay noise (exponential decay per DRL step)
            self._total_drl_steps += 1
            self._noise_std = EXPLORATION_NOISE * np.exp(
                -self._noise_decay * self._total_drl_steps
            )

        self._cached_action = action.copy()
        return action.copy()

    # ------------------------------------------------------------------
    # State augmentation  (paper eq. 7)
    # ------------------------------------------------------------------
    def build_state(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Build the 30-dim normalised DRL state.

        Components:
            x̄     = normalised  [ρ(6), v(6), w(2)]           → 14
            ū_s   = zeros (no MPC baseline for standalone)    → 3
            d̄     = normalised demand forecast (5 steps × 2)  → 10
            ū_c   = normalised previous action                → 3
                                                         Total: 30
        """
        # --- 1. Normalised freeway states (14 dims) ---
        rho = obs[:6] / RHO_MAX
        v = obs[6:12] / V_MAX
        w = np.array([
            obs[12] / W_MAX_MAIN,
            obs[13] / W_MAX_RAMP,
        ])
        x_bar = np.concatenate([rho, v, w])                     # (14,)

        # --- 2. MPC baseline output (3 dims — zeros for standalone) ---
        u_s_bar = np.zeros(3)                                    # (3,)

        # --- 3. Demand forecast (10 dims = 5 steps × 2) ---
        d_forecast = self._get_demand_forecast(step_idx)         # (10,)

        # --- 4. Previous control input (3 dims, normalised) ---
        u_c_bar = np.array([
            self._prev_action[0] / U_VSL_MAX,
            self._prev_action[1] / U_VSL_MAX,
            self._prev_action[2] / U_RAMP_MAX,
        ])                                                       # (3,)

        return np.concatenate([x_bar, u_s_bar, d_forecast, u_c_bar]).astype(np.float32)

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
    # Training
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

    def update_prev_action(self, action: np.ndarray) -> None:
        """Update the previous action (for state augmentation)."""
        self._prev_action = action.copy()

    def train_step(self) -> dict[str, float]:
        """
        One DDPG update: critic → actor → soft target update.

        Returns dict with critic_loss and actor_loss for logging.
        """
        if self.buffer.size < BATCH_SIZE:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = self.buffer.sample(BATCH_SIZE, device=self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Normalise actions to [0, 1] for critic input (Bug #6 fix)
        action_norm_t = torch.tensor(
            self._action_norm, dtype=torch.float32, device=self.device
        )
        actions_normed = actions / action_norm_t

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions_normed = next_actions / action_norm_t
            q_target = self.critic_target(next_states, next_actions_normed)
            # n-step discount: γ^n (already applied in buffer for intermediate rewards)
            gamma_n = GAMMA ** self.n_step
            y = rewards + gamma_n * (1.0 - dones) * q_target

        q_current = self.critic(states, actions_normed)
        critic_loss = F.mse_loss(q_current, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- Actor update (policy gradient) ---
        actor_actions = self.actor(states)
        actor_actions_normed = actor_actions / action_norm_t
        actor_loss = -self.critic(states, actor_actions_normed).mean()

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
    def _soft_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
        """θ' ← τ θ + (1-τ) θ'"""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(TAU * sp.data + (1.0 - TAU) * tp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save actor and critic state dicts."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load actor and critic state dicts."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
