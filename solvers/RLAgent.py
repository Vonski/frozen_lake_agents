from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RLAgent(ABC):
    """
    Abstraction for typical reinforcement learning agent. Defines interface for multiple RL
    algorithms which allows to train them with the same code.
    """
    @abstractmethod
    def get_action(self, current_state: int) -> int:
        """
        Decides what action should be taken under given circumstances.

        Args:
            current_state: State of environment at the moment.

        Returns:
            Action that should be taken.
        """
        pass

    @abstractmethod
    def learn(
        self, reward: int, action: int, current_state: int, old_state: int, **kwargs: Any
    ) -> None:
        """
        Single iteration of agent learning. It could be used either in Temporal-Difference and
        Monte Carlo settings. Calls can be differentiated by custom keyword argument.

        Args:
            reward: Reward gained as result of taken action.
            action: Taken action.
            current_state: Situation (state of environment) after action was completed.
            old_state: Situation (state of environment) in which action was taken.
            **kwargs: Additional keyword arguments. They can be used to differentiate learning
                processes between agents and algorithms.
        """
        pass

    @abstractmethod
    def copy(self) -> RLAgent:
        """
        Returns detached copy of RLAgent instance.

        Returns:
            Copy of RLAgent from which it was called.
        """
        pass

    @abstractmethod
    def save(self, filename_without_extension: str) -> None:
        """
        Saves RLAgent instance to drive.

        Args:
            filename_without_extension: Filename under which agent will be saved. Suitable
                extension will be added to it.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str) -> RLAgent:
        """
        Loads RLAgent instance from drive.

        Args:
            filename: Filename under which agent was saved.

        Returns:
            RLAgent instance read from file.
        """
        pass
