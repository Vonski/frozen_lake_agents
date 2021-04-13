from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy
import numpy as np
import torch
from neural_networks import MultiLayerPerceptron
from RLAgent import RLAgent
from torch import nn


class ReplayBuffer:
    """
    Replay buffer for collecting and batching experiences of agent.
    """

    def __init__(self, capacity: int):
        """
        Creates replay buffer with a given capacity.

        Args:
            capacity: Maximal size of the buffer.
        """
        self.experience_buffer = np.array([[0, 0, 0, 0]]).astype(int)
        self.capacity = capacity
        self.position = 0
        self.counter = 0

    def add_experience(self, experience: np.ndarray) -> None:
        """
        Adds row <S, A, R, S'> to buffer array. Description:
        S - initial state of experience.
        A - action taken by agent.
        R - reward acquired after action was taken.
        S' - new state of environment after action was taken.

        Args:
            experience: single row <S, A, R, S'>.
        """
        if self.position >= self.experience_buffer.shape[0] < self.capacity:
            self.experience_buffer = np.append(
                self.experience_buffer, experience.astype(int), axis=0
            )
        else:
            self.experience_buffer[self.position, :] = experience.astype(int)
        self.position = (self.position + 1) % self.capacity
        self.counter += 1

    def sample_experience_batch(self, batch_size: int) -> np.ndarray:
        """
        Samples batch of experiences from current buffer array. If array size is smaller than given
        `batch_size` then all rows are returned.

        Args:
            batch_size: Number of experiences that should be returned.

        Returns:
            Array of randomly chosen experiences.
        """
        batch = self.experience_buffer[
            np.random.choice(
                self.experience_buffer.shape[0],
                min(batch_size, self.experience_buffer.shape[0]),
                replace=False,
            ),
            :,
        ]
        return batch

    def get_steps_count(self) -> int:
        """
        Returns:
            Number of experiences that was ever recorded.
        """
        return self.counter


class dqnAgent(RLAgent):
    """
    Holds logic for Deep Q Network, which builds and uses neural network as an approximation
    for action-state value function in RL tasks.

    Attributes:
        local_net: Neural network representation of Q value function that is optimized.
        target_net: Neural network representation that is used to find out optimal policy Q value.
        learning_rate: Modifier of size of single learning step.
        gamma: Discount for future episodes.
        batch_size: number of experiences in a batch during single step of optimization.
        target_update_episodes: Number of episodes after which `target_net` is synchronized with
            `local_net`.
        optimizer: Torch optimizer used during training.
        loss: Torch loss used during training.
        replay_buffer: Experience replay buffer. Parametrized by 'replay_buffer_capacity' value from
            config.
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
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.gamma = kwargs.get("gamma", 0.95)
        self.batch_size = kwargs.get("batch_size", 128)
        self.target_update_episodes = kwargs.get("target_update_episodes", 4)
        self.episodes_counter = 0

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.replay_buffer = ReplayBuffer(kwargs.get("replay_buffer_capacity", 20000))

        self.local_net = MultiLayerPerceptron(state_space_size, action_space_size)
        self.optimizer = torch.optim.RMSprop(self.local_net.parameters(), lr=self.learning_rate)
        self.loss = nn.L1Loss()

        self.target_net = MultiLayerPerceptron(state_space_size, action_space_size)
        self.target_net.load_state_dict(self.local_net.state_dict())  # type: ignore
        for param in self.target_net.parameters():
            param.requires_grad = False

    def _states_to_one_hot_tensor(self, states: np.ndarray) -> torch.Tensor:
        one_hot_current_state = numpy.eye(self.state_space_size)[states]
        return torch.Tensor(one_hot_current_state).squeeze()

    def get_action(self, current_state: int) -> int:
        """
        Returns current best action based on local_net Q value approximation.

        Args:
            current_state: Current state of environment.

        Returns:
            Optimal action with respect to current Q value approximation.
        """
        tensor_one_hot_current_state = self._states_to_one_hot_tensor(np.array([current_state]))
        with torch.no_grad():
            one_hot_action_tensor = self.local_net(tensor_one_hot_current_state)
        return np.argmax(one_hot_action_tensor.numpy())

    def _compute_loss_for_experience_batch(self, experiences: np.ndarray) -> torch.Tensor:
        old_states_one_hot_encoded = self._states_to_one_hot_tensor(experiences[:, 0])
        new_states_one_hot_encoded = self._states_to_one_hot_tensor(experiences[:, 3])
        rewards_tensor = torch.Tensor(experiences[:, 2])
        actions = experiences[:, 1]

        state_action_values = self.local_net(old_states_one_hot_encoded)
        state_action_values = state_action_values[
            torch.arange(state_action_values.shape[0]), actions
        ]

        next_state_values = self.target_net(new_states_one_hot_encoded).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + rewards_tensor

        return self.loss(state_action_values, expected_state_action_values)

    def _update_target_net_if_needed(self) -> None:
        if self.episodes_counter % self.target_update_episodes == 0:
            self.target_net.load_state_dict(self.local_net.state_dict())  # type: ignore
            for param in self.target_net.parameters():
                param.requires_grad = False

    def learn(
        self, reward: int, action: int, current_state: int, old_state: int, **kwargs: Any
    ) -> None:
        """
        Experience collection and single `local_net` optimization step at the end of episode. It
        also updates `target_net` if it is right time.

        Args:
            reward: Reward acquired after taking `action` in `old_state`
            action: Action taken by agent.
            current_state: New state of environment.
            old_state: Previous state of environment.
            **kwargs: additional keywords arguments. E.g. `end_of_episode` which tells that
                `current_state` is terminal.
        """
        if "end_of_episode" in kwargs:
            experiences = self.replay_buffer.sample_experience_batch(self.batch_size)
            loss = self._compute_loss_for_experience_batch(experiences)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.episodes_counter += 1
            self._update_target_net_if_needed()
        else:
            self.replay_buffer.add_experience(
                np.array([[old_state, action, reward, current_state]])
            )

    def copy(self) -> dqnAgent:
        """
        Creates identical copy of agent that this method was called from.

        Returns:
            New agent.
        """
        return deepcopy(self)

    def save(self, filename_without_extension: str) -> None:
        """
        Saves agent object data to drive.

        Args:
            filename_without_extension: Name of file in which data will be saved.
        """
        torch.save(self, filename_without_extension + ".pt")

    @staticmethod
    def load(filename: str) -> dqnAgent:
        """
        Create new agent from data loaded from drive.

        Args:
            filename: Full name of file in which agent object data is stored.

        Returns:
            New agent.
        """
        new_agent = torch.load(filename)
        return new_agent
