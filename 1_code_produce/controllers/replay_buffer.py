"""
N-step TD Replay Buffer for DDPG.

Stores transitions in order and supports sampling n-step returns.
For n-step TD, each sampled transition bundles:
    (s_i, a_i, R_n_i, s_{i+n}, done_{i+n})
where  R_n = Σ_{j=0}^{n-1} γ^j r_{i+j}.
"""

import numpy as np
import torch


class NStepReplayBuffer:
    """
    Ring buffer with n-step return computation.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    n_step : int
        Number of steps for multi-step TD (1 = standard, 10 = 10-step).
    gamma : float
        Discount factor for computing n-step returns.
    """

    def __init__(self, capacity: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []

        self._pos = 0       # write position (ring buffer)
        self._size = 0      # current number of stored transitions
        self._full = False   # whether buffer has wrapped around

    @property
    def size(self) -> int:
        """Number of *usable* transitions (accounting for n-step lookahead)."""
        return max(0, self._size - self.n_step + 1)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to the buffer."""
        if self._full:
            self.states[self._pos] = state
            self.actions[self._pos] = action
            self.rewards[self._pos] = reward
            self.next_states[self._pos] = next_state
            self.dones[self._pos] = done
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        if self._size == self.capacity:
            self._full = True

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Sample a mini-batch of n-step transitions.

        Returns dict with keys: states, actions, rewards, next_states, dones.
        - rewards: n-step discounted return  R_n = Σ γ^j r_{i+j}
        - next_states: the state n steps ahead (s_{i+n})
        - dones: whether a terminal state was reached within the n steps
        """
        usable = self.size
        assert usable >= batch_size, (
            f"Not enough transitions: {usable} < {batch_size}"
        )

        # Valid starting indices: must have n consecutive transitions
        # that don't cross an episode boundary (done flag)
        if self._full:
            valid_indices = []
            for i in range(self._size):
                end_idx = i + self.n_step - 1
                if end_idx >= self._size:
                    continue
                # Check that no intermediate transition (except the last)
                # has done=True, which would indicate an episode boundary
                ok = True
                for j in range(self.n_step - 1):
                    if self.dones[i + j]:
                        ok = False
                        break
                if ok:
                    valid_indices.append(i)
            valid_indices = np.array(valid_indices)
        else:
            max_start = self._size - self.n_step
            valid_indices = []
            for i in range(max_start + 1):
                ok = True
                for j in range(self.n_step - 1):
                    if self.dones[i + j]:
                        ok = False
                        break
                if ok:
                    valid_indices.append(i)
            valid_indices = np.array(valid_indices)

        chosen = np.random.choice(valid_indices, size=batch_size, replace=True)

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        gamma_powers = np.array([self.gamma ** j for j in range(self.n_step)])

        for i in chosen:
            batch_states.append(self.states[i])
            batch_actions.append(self.actions[i])

            # Compute n-step return and find the effective next state
            R_n = 0.0
            effective_next = self.next_states[i]
            effective_done = False
            for j in range(self.n_step):
                idx = (i + j) % self.capacity if self._full else i + j
                R_n += gamma_powers[j] * self.rewards[idx]
                effective_next = self.next_states[idx]
                if self.dones[idx]:
                    effective_done = True
                    break  # episode ended; don't look further

            batch_rewards.append(R_n)
            batch_next_states.append(effective_next)
            batch_dones.append(effective_done)

        return {
            "states": torch.tensor(
                np.array(batch_states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(
                np.array(batch_actions), dtype=torch.float32, device=device
            ),
            "rewards": torch.tensor(
                batch_rewards, dtype=torch.float32, device=device
            ).unsqueeze(1),
            "next_states": torch.tensor(
                np.array(batch_next_states), dtype=torch.float32, device=device
            ),
            "dones": torch.tensor(
                batch_dones, dtype=torch.float32, device=device
            ).unsqueeze(1),
        }

    def clear(self) -> None:
        """Remove all transitions."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self._pos = 0
        self._size = 0
        self._full = False
