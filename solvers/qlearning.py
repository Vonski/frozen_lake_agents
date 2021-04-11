from __future__ import annotations

from typing import Any

import numpy as np
from RLAgent import RLAgent


class QLearningAgent(RLAgent):
    def __init__(self, state_space_size: int, action_space_size: int, **kwargs: Any):
        self.quality_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = kwargs.get("learning_rate", 0.9)
        self.gamma = kwargs.get("gamma", 0.9)

    def get_action(self, current_state: int) -> int:
        return np.argmax(self.quality_table[current_state, :])

    def learn(
        self, reward: int, action: int, current_state: int, old_state: int, **kwargs: Any
    ) -> None:
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
        return self.__class__._from_quality_table(self.quality_table)

    def save(self, filename_without_extension: str) -> None:
        np.save(filename_without_extension, self.quality_table)

    @staticmethod
    def load(filename: str) -> QLearningAgent:
        quality_table = np.load(filename)
        return QLearningAgent._from_quality_table(quality_table)
