"""
DDPG Actor and Critic networks for Sun et al. (2024) replication.

Actor:  30 → 256 → 256 → 3  (ReLU, sigmoid-scaled output)
Critic: state(30 → 256) ⊕ action(3 → 128) → 256 → 128 → 1
"""

import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """
    Deterministic policy  π_θ : S → A.

    Architecture (from Section IV-B.2):
        Input(30) → 256 → 256 → Output(3)
        Hidden activations: ReLU
        Output: sigmoid scaled to [action_low, action_high]
    """

    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 3,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # Default action bounds: [20, 20, 0] – [102, 102, 1]
        if action_low is None:
            action_low = np.array([20.0, 20.0, 0.0])
        if action_high is None:
            action_high = np.array([102.0, 102.0, 1.0])

        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action in [action_low, action_high]."""
        raw = torch.sigmoid(self.net(state))
        return self.action_low + raw * (self.action_high - self.action_low)


class Critic(nn.Module):
    """
    Q-value function  Q_φ(s, a) → ℝ.

    Architecture (from Section IV-B.2):
        State branch:  30 → 256  (ReLU)
        Action branch: 3  → 128  (ReLU)
        Merged:        384 → 256 → 128 → 1
        Hidden activations: ReLU
        Output: linear (no activation)
    """

    def __init__(self, state_dim: int = 30, action_dim: int = 3):
        super().__init__()
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        self.action_branch = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
        )
        self.merged = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-value for (state, action) pair."""
        s = self.state_branch(state)
        a = self.action_branch(action)
        return self.merged(torch.cat([s, a], dim=-1))
