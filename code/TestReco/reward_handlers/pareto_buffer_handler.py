from typing import Any, List, Optional, Sequence

import numpy as np


class ParetoBufferHandler:
    """
    Handler for maintaining a Pareto-optimal buffer of trajectories for multi-objective RL.
    Can be combined with other reward handlers (scalarized or reward machine).
    """

    def __init__(self, buffer_size: int = 10, base_handler: Optional[Any] = None):
        self.buffer_size = buffer_size
        self.buffer: List[Any] = []  # Each item is a trajectory (list of steps)
        self.base_handler = (
            base_handler  # ScalarizedRewardHandler or RewardMachineHandler
        )

    def process_reward(self, reward_dict: dict, **kwargs):
        """
        Process reward using the base handler if available, otherwise return the reward dictionary.
        """
        if self.base_handler is not None:
            # Pass all kwargs to the base handler
            return self.base_handler.process_reward(reward_dict, **kwargs)
        else:
            return reward_dict

    def add_trajectory(self, trajectory: List[Any]):
        """
        Add a trajectory (list of steps, each with a reward vector) to the buffer, keeping only Pareto-optimal ones.
        """
        self.buffer.append(trajectory)
        self._update_pareto_buffer()
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[: self.buffer_size]

    def _update_pareto_buffer(self):
        # Only keep Pareto-optimal (non-dominated) trajectories
        pareto_front = []
        for traj in self.buffer:
            dominated = False
            traj_reward = self._trajectory_objective(traj)
            for other in self.buffer:
                if other is traj:
                    continue
                other_reward = self._trajectory_objective(other)
                if self._dominates(other_reward, traj_reward):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(traj)
        self.buffer = pareto_front

    def _trajectory_objective(self, traj: List[Any]) -> np.ndarray:
        # Sum rewards across the trajectory (assume each step has a 'reward' field)
        # Handle dictionary format for rewards
        if traj and "reward" in traj[0]:
            first_step = traj[0]
            if "reward_dict" in first_step["reward"]:
                # Dictionary format: convert to array based on objectives order
                objectives = list(first_step["reward"]["reward_dict"].keys())
                return np.sum([np.array([step["reward"]["reward_dict"][obj] for obj in objectives]) for step in traj], axis=0)
            else:
                raise ValueError("reward_dict not found in step reward")
        raise ValueError("Reward not found in trajectory")

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        # Returns True if a dominates b (all a >= b and at least one a > b)
        return np.all(a >= b) and np.any(a > b)

    def sample_batch(self, batch_size: int) -> List[Any]:
        # Randomly sample trajectories from the buffer
        idxs = np.random.choice(
            len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False
        )
        return [self.buffer[i] for i in idxs]
