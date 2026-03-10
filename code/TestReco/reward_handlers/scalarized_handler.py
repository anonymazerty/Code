from typing import Any, Sequence

import numpy as np


class ScalarizedRewardHandler:
    """
    Handler for scalarizing multi-objective reward vectors using a weighted sum.
    """

    def __init__(self, weights: Sequence[float], objectives: Sequence[str]):
        self.weights = np.array(weights)
        self.objectives = objectives

    def process_reward(self, reward_dict: dict, **kwargs) -> float:
        """
        Compute the scalarized reward as a weighted sum.
        Args:
            reward_dict: Dictionary mapping objective names to reward values
        Returns:
            Scalarized reward (float)
        """
        # Convert dictionary to array based on weights order
        reward_vec = np.array([reward_dict[obj] for obj in self.objectives])
        return float(np.dot(self.weights, reward_vec))
