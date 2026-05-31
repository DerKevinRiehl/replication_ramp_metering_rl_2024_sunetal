"""
No-control baseline: always applies free-flow speed and full ramp metering.
"""

import numpy as np
from controllers.base_controller import Controller


class NoControlController(Controller):
    """
    Passive controller – no traffic management.

    Sets VSL to free-flow speed and ramp metering rate to 1.0 at every step.
    """

    def __init__(self, v_free: float = 102.0):
        self._v_free = v_free
        self._action = np.array([v_free, v_free, 1.0], dtype=np.float32)

    @property
    def update_interval(self) -> int:
        return 1  # trivially, action never changes

    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        return self._action.copy()

    def reset(self) -> None:
        pass  # nothing to reset
