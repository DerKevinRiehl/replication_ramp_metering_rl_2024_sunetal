"""
METANET Traffic Simulation Environment.

Gym wrapper around a CasADi-based METANET model.
Each call to step() executes exactly ONE T_s = 10 s simulation step.
Multi-step orchestration (T_d = 60 s for DRL, T_c = 300 s for MPC)
is handled externally by the controller / main loop.

Observation : x = [rho(6); v(6); w(2)]  (14-dim, flattened)
Action      : [VSL1_km/h, VSL2_km/h, ramp_metering_rate]
"""

import gymnasium as gym
import numpy as np
import casadi as cs
from gymnasium import spaces

from config import TS
from sym_metanet import engines
from metanet.metanet_utils import warmup_metanet
from metanet.metanet_model import build_network, build_dynamics_function


class MetanetEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Gym wrapper around the CasADi-based METANET model.

    Each call to ``step(action)`` advances the simulation by exactly
    T_s = 10 s (one METANET update).  Higher-level control frequencies
    (T_d = 60 s for DRL, T_c = 300 s for MPC) are handled by the
    controller that drives this environment.

    Parameters
    ----------
    metanet_param : dict
        Physical parameters (e.g. METANET_PARAMS["real"]).
    demand : np.ndarray, shape (n_steps, 2)
        Demand profile [mainstream_veh/h, ramp_veh/h] per sim step.
    noise_cfg : dict
        Gaussian noise config with keys "main" and "ramp", each having
        "mean" and "std".
    seed : int
        Random seed.
    """

    metadata = {"render.modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, metanet_param: dict, demand, noise_cfg: dict, seed: int = 0):
        super().__init__()

        self.metanet_param = metanet_param
        self.demand = np.array(demand)          # (n_steps, 2)
        self.noise_cfg = noise_cfg
        self.n_steps = self.demand.shape[0]

        # Build CasADi network & dynamics function
        engines.use("casadi", sym_type="SX")
        self.net = build_network(metanet_param)
        self.F = build_dynamics_function(self.net, metanet_param)

        # Warm-up state (computed once)
        self.x_warm = warmup_metanet(self.F, self.net, self.metanet_param)

        # Dimensions
        self.n_segments = sum(link.N for _, _, link in self.net.links)
        self.n_origins = len(self.net.origins)

        obs_dim = 2 * self.n_segments + self.n_origins   # rho + v + w
        self.obs_dim = obs_dim

        # Convenience aliases for external code
        self.ns = obs_dim                                 # state dim
        self.na = 3                                       # action dim
        self.nd = self.demand.shape[1]                    # disturbance dim

        # Action bounds
        v_min, v_max = 20.0, metanet_param["v_free_1"]
        low  = np.array([v_min, v_min, 0.0], dtype=np.float32)
        high = np.array([v_max, v_max, 1.0], dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Internal state
        self.rng = np.random.RandomState(seed)
        self.t = 0                          # current sim-step index
        self.x: cs.DM | None = None         # CasADi state vector
        self.last_action = np.array([v_max, v_max, 1.0], dtype=np.float32)

        # Per-step logging
        self.log_rho: list[np.ndarray] = []
        self.log_v:   list[np.ndarray] = []
        self.log_w:   list[np.ndarray] = []

        # Episode-level statistics
        self.ep_tts = 0.0
        self.ep_twt = 0.0
        self.ep_queue_violation = 0.0
        self.ep_min_speed = np.inf

        # History across episodes (useful during training)
        self.ep_tts_history:             list[float] = []
        self.ep_twt_history:             list[float] = []
        self.ep_queue_violation_history: list[float] = []
        self.ep_min_speed_history:       list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state_to_obs(self, x_dm: cs.DM) -> np.ndarray:
        return np.array(x_dm).reshape(-1).astype(np.float32)

    def parse_obs(self, obs: np.ndarray):
        """Split a flat observation into (rho, v, w, _, _)."""
        rho = obs[: self.n_segments]
        v   = obs[self.n_segments : 2 * self.n_segments]
        w   = obs[2 * self.n_segments : 2 * self.n_segments + self.n_origins]
        return rho, v, w, None, None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        super().reset(seed=seed)

        self.x = cs.DM(self.x_warm)
        self.t = 0
        self.last_action = self.action_space.high.copy()

        self.log_rho.clear()
        self.log_v.clear()
        self.log_w.clear()

        self.ep_tts = 0.0
        self.ep_twt = 0.0
        self.ep_queue_violation = 0.0
        self.ep_min_speed = np.inf

        return self._state_to_obs(self.x), {}

    def step(self, action: np.ndarray):
        """
        Advance the simulation by exactly **one** T_s = 10 s step.

        Parameters
        ----------
        action : array-like, shape (3,)
            [VSL1_km/h, VSL2_km/h, ramp_metering_rate]

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        a = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        T_hr = TS / 3600.0

        # --- Noisy disturbance for this step ---
        d_k = self.demand[self.t, :].copy()
        d_k[0] += self.rng.normal(self.noise_cfg["main"]["mean"],
                                  self.noise_cfg["main"]["std"])
        d_k[1] += self.rng.normal(self.noise_cfg["ramp"]["mean"],
                                  self.noise_cfg["ramp"]["std"])

        # --- METANET dynamics (CasADi) ---
        d_dm = cs.DM(d_k.tolist())
        u_dm = cs.DM(a.tolist())
        x_next, _q_all = self.F(self.x, u_dm, d_dm)
        self.x = cs.DM(cs.fmax(0.0, x_next))           # enforce ≥ 0

        # --- Extract state components ---
        x_np = np.array(self.x).reshape(-1)
        rho = x_np[: self.n_segments]
        v   = x_np[self.n_segments : 2 * self.n_segments]
        w   = x_np[2 * self.n_segments : 2 * self.n_segments + self.n_origins]

        # --- Logging ---
        self.log_rho.append(rho.copy())
        self.log_v.append(v.copy())
        self.log_w.append(w.copy())

        # --- TTS / TWT computation for this single step ---
        veh_in_network = (rho * self.metanet_param["L_main"]
                          * self.metanet_param["lambda_1"]).sum()
        queues_all = w.sum()

        tts = T_hr * (veh_in_network + queues_all)
        twt = T_hr * queues_all

        # Queue constraint violations
        violation_main = max(0.0, w[0] - self.metanet_param["queue_max_main"])
        violation_ramp = max(0.0, w[1] - self.metanet_param["queue_max_ramp"])
        penalty = T_hr * (violation_main + violation_ramp)

        # Queue violation ratio: max exceeded / max allowed (for reporting)
        ratio_main = violation_main / self.metanet_param["queue_max_main"]
        ratio_ramp = violation_ramp / self.metanet_param["queue_max_ramp"]
        max_ratio = max(ratio_main, ratio_ramp) * 100.0  # percentage

        # --- Update episode-level accumulators ---
        self.ep_tts += tts
        self.ep_twt += twt
        self.ep_queue_violation = max(self.ep_queue_violation, max_ratio)
        self.ep_min_speed = min(self.ep_min_speed, float(v.min()))

        # --- Advance time ---
        self.t += 1
        terminated = self.t >= self.n_steps
        truncated = False

        # --- Reward (negative TTS + smoothness penalty) ---
        du = a - self.last_action
        smooth_pen = 0.4 * float(np.dot(du, du))
        reward = -(tts + smooth_pen + 10.0 * penalty)

        self.last_action = a.copy()

        # --- Info dict ---
        info = {
            "tts_step":             float(tts),
            "twt_step":             float(twt),
            "queue_violation_step": float(penalty),
            "min_speed_step":       float(v.min()),
        }

        # Archive episode stats when finished
        if terminated or truncated:
            self.ep_tts_history.append(self.ep_tts)
            self.ep_twt_history.append(self.ep_twt)
            self.ep_queue_violation_history.append(self.ep_queue_violation)
            self.ep_min_speed_history.append(self.ep_min_speed)

        return self._state_to_obs(self.x), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Render / repr
    # ------------------------------------------------------------------
    def render(self, mode="human"):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.__class__.__name__}()"