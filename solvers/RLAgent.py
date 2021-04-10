from abc import ABC, abstractmethod


class RLAgent(ABC):
    @abstractmethod
    def get_action(self, current_state):
        pass

    @abstractmethod
    def learn(self, reward, action, current_state, old_state, **kwargs):
        pass

    @abstractmethod
    def save(self, filename_without_extension):
        pass

    @staticmethod
    @abstractmethod
    def load(filename):
        pass
