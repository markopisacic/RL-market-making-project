from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Abstract class representing classic environment in reinforcement learning problems. Other classes extend this one.
    """

    def __init__(self):
        super(BaseEnv, self).__init__()

    @abstractmethod
    def step(self, action):
        """Execute one step within the environment, with the given action."""
        pass

    @abstractmethod
    def reset(self):
        """Set the environment to initial configuration."""
        pass
