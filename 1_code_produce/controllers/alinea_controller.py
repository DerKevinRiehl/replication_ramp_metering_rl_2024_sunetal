"""
ALINEA ramp metering controller.

Classic feedback controller (Papageorgiou et al., 1991):
    r(k) = r(k-1) + K_R * (rho_crit - rho_downstream)

Only controls the on-ramp metering rate; VSL is kept at free-flow speed.
"""

import numpy as np
from controllers.base_controller import Controller


class AlineaController(Controller):
    """
    Standalone ALINEA ramp metering controller.

    Parameters
    ----------
    v_free : float
        Free-flow speed [km/h] (used for VSL passthrough).
    rho_crit : float
        Critical density [veh/km/lane] of the downstream segment.
    K_R : float
        ALINEA regulator gain [km/lane]. Default 70 is a standard
        value from the Papageorgiou literature.
    downstream_seg_idx : int
        0-based index of the segment immediately downstream of the
        on-ramp in the observation vector.  In the 6-segment METANET
        network the ramp merges between segments 2 and 3, so the
        downstream segment is index 3.
    n_segments : int
        Number of segments in the METANET network (needed to locate
        the density slice inside the flat observation vector).
    update_steps : int
        Number of T_s steps between control re-computations.
        Default 6 (= 60 s, matching DRL frequency).
    """

    def __init__(
        self,
        v_free: float = 102.0,
        rho_crit: float = 33.5,
        K_R: float = 70.0,
        downstream_seg_idx: int = 3,
        n_segments: int = 6,
        update_steps: int = 6,
    ):
        self._v_free = v_free
        self._rho_crit = rho_crit
        self._K_R = K_R
        self._ds_idx = downstream_seg_idx
        self._n_seg = n_segments
        self._update_steps = update_steps

        # Internal state
        self._r: float = 1.0           # current metering rate
        self._action: np.ndarray = np.array(
            [v_free, v_free, 1.0], dtype=np.float32
        )

    # ------------------------------------------------------------------
    #  Controller ABC interface
    # ------------------------------------------------------------------
    @property
    def update_interval(self) -> int:
        return self._update_steps

    def reset(self) -> None:
        self._r = 1.0
        self._action = np.array(
            [self._v_free, self._v_free, 1.0], dtype=np.float32
        )

    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Return [v_free, v_free, r] where r is updated by ALINEA feedback.

        Re-computes r only every ``update_steps`` simulation steps;
        otherwise returns the cached action.
        """
        if step_idx % self._update_steps != 0:
            return self._action.copy()

        # Downstream density: obs is [rho(n_seg), v(n_seg), w(n_orig)]
        rho_down = float(obs[self._ds_idx])

        # ALINEA update:  r(k) = r(k-1) + K_R * (rho_crit - rho_down)
        self._r = self._r + self._K_R * (self._rho_crit - rho_down)
        self._r = float(np.clip(self._r, 0.0, 1.0))

        self._action = np.array(
            [self._v_free, self._v_free, self._r], dtype=np.float32
        )
        return self._action.copy()
