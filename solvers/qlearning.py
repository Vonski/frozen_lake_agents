from __future__ import annotations

from typing import Any

import numpy as np
from RLAgent import RLAgent


class QLearningAgent(RLAgent):
    """
    Holds logic for Q learning algorithm, which builds and uses lookup table as an approximation
    for action-state value function in RL tasks.

    Attributes:
        quality_table: Lookup table for action-state value function.
        learning_rate: Modifier of size of single learning step.
        gamma: Discount for future episodes.
    """

    def __init__(self, state_space_size: int, action_space_size: int, **kwargs: Any):
        """
        Creates agent that can operate in environment with state and actions spaces of given sizes.
        Additional configuration arguments can be provided through `kwargs`.

        Args:
            state_space_size: Size of state space.
            action_space_size: Size of action space.
            **kwargs: Additional configuration arguments specific for this type of agent.
        """
        self.quality_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = kwargs.get("learning_rate", 0.9)
        self.gamma = kwargs.get("gamma", 0.9)

    def get_action(self, current_state: int) -> int:
        """
        Returns current best action based on quality table.

        Args:
            current_state: Current state of environment.

        Returns:
            Optimal action with respect to quality table.
        """
        return np.argmax(self.quality_table[current_state, :])

    def learn(
        self, reward: int, action: int, current_state: int, old_state: int, **kwargs: Any
    ) -> None:
        """
        Single quality table update.

        Args:
            reward: Reward acquired after taking `action` in `old_state`
            action: Action taken by agent.
            current_state: New state of environment.
            old_state: Previous state of environment.
            **kwargs: additional keywords arguments. E.g. `end_of_episode` which tells that
                `current_state` is terminal.
        """
        if "end_of_episode" in kwargs:
            return

        self.quality_table[old_state, action] += self.learning_rate * (
            reward
            + self.gamma * np.max(self.quality_table[current_state, :])
            - self.quality_table[old_state, action]
        )

    @staticmethod
    def _from_quality_table(quality_table: np.ndarray) -> QLearningAgent:
        new_agent = QLearningAgent(quality_table.shape[1], quality_table.shape[0])
        new_agent.quality_table = quality_table
        return new_agent

    def copy(self) -> QLearningAgent:
        """
        Creates new agent with identical quality table.

        Returns:
            New agent.
        """
        return self.__class__._from_quality_table(self.quality_table)

    def save(self, filename_without_extension: str) -> None:
        """
        Saves quality table to drive.

        Args:
            filename_without_extension: Name of file in which quality table will be saved.
        """
        np.save(filename_without_extension, self.quality_table)

    @staticmethod
    def load(filename: str) -> QLearningAgent:
        """
        Create new agent from quality table loaded from drive.

        Args:
            filename: Full name of file in which quality table is stored.

        Returns:
            New agent.
        """
        quality_table = np.load(filename)
        return QLearningAgent._from_quality_table(quality_table)
