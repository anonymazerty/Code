from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BasePolicy(ABC):
    """
    Base class for all policies.
    """

    def __init__(self):
        """Initialize the policy."""
        pass

    @abstractmethod
    def select_action(
        self, state: np.ndarray, available_actions: Optional[List[int]] = None, **kwargs
    ) -> Tuple[int, Dict]:
        """
        Select an action based on the current state.

        Args:
            state: Current state
            available_actions: List of available actions (optional)
            **kwargs: Additional arguments

        Returns:
            Tuple of (action, info_dict)
        """
        pass

    @abstractmethod
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        **kwargs
    ) -> float:
        """
        Update the policy using a batch of experiences.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the policy parameters."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the policy parameters."""
        pass
