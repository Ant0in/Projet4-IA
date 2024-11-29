
from .env import Labyrinth
import numpy as np
import random
from collections import defaultdict


class QLearning:

    """
    Q-Learning algorithm for training an agent in a given environment.
    The agent learns an optimal policy for selecting actions to maximize cumulative rewards.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    - alpha (float): Learning rate.
    - epsilon (float): Probability of taking a random action (exploration).
    - c (float): Parameter for exploration/exploitation balance in action selection.
    ...
    """

    def __init__(self, env: Labyrinth, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0, c: float = 0):
        """
        Initialize the Q-Learning agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        - alpha (float): Learning rate (0 < alpha <= 1) for updating Q-values.
        - epsilon (float): Probability (0 <= epsilon <= 1) for exploration in action selection.
        - c (float): Exploration adjustment parameter.
        """
        self.env = env
        self.gamma = gamma          
        self.alpha = alpha          
        self.epsilon = epsilon      
        self.c = c                  
        

    def get_q_table(self) -> np.ndarray:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """
        raise NotImplementedError()


    def train(self, n_steps: int):
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """
        raise NotImplementedError()