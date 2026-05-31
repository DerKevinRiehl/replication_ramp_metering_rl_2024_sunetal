"""
Controller wrapper for evaluating the combined MPC-DRL framework.

This class couples:
  1) an MPC baseline controller (updated every T_c = 300 s), and
  2) a trained DDPG-MPC agent that computes corrective actions every
     T_d = 60 s.

At each DRL step:
    u_c = clip(u_b + u_rl, U_min, U_max)
where u_b is the latest MPC action and u_rl is the DRL adjustment.
"""

import os
import numpy as np

from controllers.base_controller import Controller
from controllers.mpc_controller import MPCController
from controllers.ddpg_mpc_agent import DDPGMPCAgent
from config import DDPG_STEP


class MPCDRLController(Controller):
    """Inference controller for the combined MPC-DRL architecture."""

    def __init__(
        self,
        env,
        demand: np.ndarray,
        mpc_params: dict,
        n_step: int = 10,
        model_dir: str | None = None,
        model_tag: str | None = None,
        model_type: str = "best",
        device: str = "cpu",
        **mpc_kwargs,
    ):
        self.demand = np.asarray(demand)
        self.n_step = n_step

        self.mpc = MPCController(
            env=env,
            demand=demand,
            mpc_params=mpc_params,
            **mpc_kwargs,
        )
        self.agent = DDPGMPCAgent(
            demand=demand,
            estimation_step=n_step,
            device=device,
        )

        self._cached_action = np.array(
            [mpc_params["v_free_1"], mpc_params["v_free_1"], 1.0],
            dtype=np.float32,
        )
        self._last_mpc_action = self._cached_action.copy()

        if model_dir is not None and model_tag is not None:
            load_path = os.path.join(model_dir, f"{model_tag}_{model_type}.pt")
            if os.path.isfile(load_path):
                self.agent.load(load_path)
                print(f"  [MPC-DRL] Loaded model from {load_path}")
            else:
                print(f"  [MPC-DRL] WARNING: No model found at {load_path}")

    @property
    def update_interval(self) -> int:
        return int(DDPG_STEP)

    def reset(self) -> None:
        self.mpc.reset()
        self.agent.reset()
        self._cached_action = np.array(
            [self.mpc.v_free, self.mpc.v_free, 1.0],
            dtype=np.float32,
        )
        self._last_mpc_action = self._cached_action.copy()

    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        # Reuse the last control between DRL decision instants.
        if step_idx % self.update_interval != 0:
            return self._cached_action.copy()

        # Refresh baseline MPC action only at MPC decision instants.
        if step_idx % self.mpc.update_interval == 0:
            self._last_mpc_action = self.mpc.get_action(obs, step_idx).astype(
                np.float32
            )

        state = self.agent.build_state(obs, step_idx, self._last_mpc_action)
        u_rl = self.agent.select_action(state, add_noise=False)
        u_c = self.agent.saturate(self._last_mpc_action, u_rl)

        self.agent.update_prev_action(u_c)
        self._cached_action = u_c.copy()
        return u_c.copy()
