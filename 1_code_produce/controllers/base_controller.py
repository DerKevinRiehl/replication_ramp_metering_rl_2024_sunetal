"""
Abstract base class for all traffic controllers.

Every controller implements:
 - ``update_interval``: how many T_s steps between re-computations
 - ``get_action(obs, step_idx)``: returns the action to apply at step_idx
 - ``reset()``: resets any internal state between episodes
"""

from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """Base class for freeway traffic controllers."""

    @property
    @abstractmethod
    def update_interval(self) -> int:
        """Number of T_s steps between control re-computations."""
        ...

    @abstractmethod
    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Return the control action for the current simulation step.

        The controller internally decides whether to recompute or return
        a cached action based on ``step_idx % update_interval``.

        Parameters
        ----------
        obs : np.ndarray
            Current environment observation.
        step_idx : int
            Current simulation-step index (0-based).

        Returns
        -------
        action : np.ndarray, shape (3,)
            [VSL1_km/h, VSL2_km/h, ramp_metering_rate]
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (called at the start of each episode)."""
        ...
