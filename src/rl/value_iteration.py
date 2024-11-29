

import numpy as np
from .env import Labyrinth
from collections import defaultdict
from lle import Action



class ValueIteration:

    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    ...
    """
    
    def __init__(self, env: Labyrinth, gamma: float = 1.0):

        self.env: Labyrinth = env
        self.gamma: float = gamma
        self.value_table: np.ndarray = np.zeros(env.get_map_size())


    def train(self, n_updates: int):

        for nth_update in range(n_updates):
            print(f'Update : {nth_update}')


    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        return self.value_table