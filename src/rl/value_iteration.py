import numpy as np
from .env import Labyrinth
from collections import defaultdict
class ValueIteration:
    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    ...
    """
    
    def __init__(self, env: Labyrinth, gamma: float = 1):
        """
        Initialize the Value Iteration agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        """
        self.env = env
        self.gamma = gamma  

    def train(self,  n_updates: int):
        """
        Train the agent using value iteration for a specified number of updates.

        Parameters:
        - n_updates (int): The total number of updates to perform.
        """
        raise NotImplementedError()


    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        raise NotImplementedError()