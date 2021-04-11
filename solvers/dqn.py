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
    def __init__(self, capacity: int, success_ratio_target: float = 0.0):
        self.experience_buffer = np.array([[0, 0, 0, 0]]).astype(int)
        self.capacity = capacity
        self.success_ratio_target = success_ratio_target
        self.position = 0
        self.success_ratio = 0.0

    def add_experience(self, experience: np.ndarray) -> None:
        if experience[0, 2] == 1:
            self.success_ratio += 1 / self.capacity
        if self.position >= self.experience_buffer.shape[0] < self.capacity:
            self.experience_buffer = np.append(
                self.experience_buffer, experience.astype(int), axis=0
            )
        else:
            while (
                self.success_ratio < self.success_ratio_target
                and self.experience_buffer[self.position, 2] == 1
            ):
                self.position = (self.position + 1) % self.capacity
            if self.experience_buffer[self.position, 2] == 1:
                self.success_ratio -= 1 / self.capacity
            self.experience_buffer[self.position, :] = experience.astype(int)
        self.position = (self.position + 1) % self.capacity

    def sample_experience_batch(self, batch_size: int) -> np.ndarray:
        batch = self.experience_buffer[
            np.random.choice(
                self.experience_buffer.shape[0],
                min(batch_size, self.experience_buffer.shape[0]),
                replace=False,
            ),
            :,
        ]
        return batch

    def get_experience_count(self) -> int:
        return self.experience_buffer.shape[0]


class dqnAgent(RLAgent):
    def __init__(self, state_space_size: int, action_space_size: int, **kwargs: Any):
        self.learning_rate = kwargs.get("learning_rate", 0.7)
        self.gamma = kwargs.get("gamma", 0.9)
        self.batch_size = kwargs.get("batch_size", 128)
        self.steps_per_epoch = kwargs.get("steps_per_epoch", 5000)
        self.epoch = 0

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.replay_buffer = ReplayBuffer(self.steps_per_epoch, kwargs.get("success_ratio", 0.1))

        self.local_net = MultiLayerPerceptron(state_space_size, action_space_size)
        self.optimizer = torch.optim.RMSprop(self.local_net.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss(reduction="sum")

        self.target_net = MultiLayerPerceptron(state_space_size, action_space_size)
        self.target_net.load_state_dict(self.local_net.state_dict())  # type: ignore
        for param in self.target_net.parameters():
            param.requires_grad = False

    def _states_to_one_hot_tensor(self, states: np.ndarray) -> torch.Tensor:
        one_hot_current_state = numpy.eye(self.state_space_size)[states]
        return torch.Tensor(one_hot_current_state).squeeze()

    def get_action(self, current_state: int) -> int:
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
        if self.replay_buffer.get_experience_count() // self.steps_per_epoch > self.epoch:
            self.epoch = self.replay_buffer.get_experience_count() // self.steps_per_epoch
            self.target_net.load_state_dict(self.local_net.state_dict())  # type: ignore
            for param in self.target_net.parameters():
                param.requires_grad = False

    def learn(
        self, reward: int, action: int, current_state: int, old_state: int, **kwargs: Any
    ) -> None:
        if "end_of_episode" in kwargs:
            experiences = self.replay_buffer.sample_experience_batch(self.batch_size)
            loss = self._compute_loss_for_experience_batch(experiences)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.replay_buffer.add_experience(
                np.array([[old_state, action, reward, current_state]])
            )
        self._update_target_net_if_needed()

    def copy(self) -> dqnAgent:
        return deepcopy(self)

    def save(self, filename_without_extension: str) -> None:
        torch.save(self, filename_without_extension + ".pt")

    @staticmethod
    def load(filename: str) -> dqnAgent:
        new_agent = torch.load(filename)
        return new_agent
