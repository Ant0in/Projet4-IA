
from .env import Labyrinth
import numpy as np
import random
from collections import defaultdict


# alias
State = tuple[int, int]


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

    def __init__(self, env: Labyrinth, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.0, c: float = 0.0) -> None:
        
        """
        Initialize the Q-Learning agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        - alpha (float): Learning rate (0 < alpha <= 1) for updating Q-values.
        - epsilon (float): Probability (0 <= epsilon <= 1) for exploration in action selection.
        - c (float): Exploration adjustment parameter.
        """

        self.env: Labyrinth = env
        self.gamma: float = gamma          
        self.alpha: float = alpha          
        self.epsilon: float = epsilon     

        # Pour l'exploration et le bonus, on garde un defaultdict des states-actions.
        self.c: float = c
        self.explored_dict: defaultdict = defaultdict(int)

        self.possible_states: list[State] = env.get_valid_states()
        # Créer la table de dimension 3 (chaques states (2d) associés aux actions (1d))
        self.qtable: np.ndarray = np.zeros((env.get_map_size() + (len(env.get_all_actions()),)))
        

    def get_q_table(self) -> np.ndarray:
        
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """

        return self.qtable
    
    def get_state_value(self, s: State) -> float:
        # Get state value in the qvalue table
        return self.get_q_table()[s[0], s[1]]
    
    def set_state_value(self, s: State, action_id: int, value: float) -> None:
        # Set state value in the qvalue table
        self.get_q_table()[s[0], s[1], action_id] = value

    def get_state_action_exploration_number(self, s: State, action_id: int) -> int:
        return self.explored_dict[(s, action_id)]

    def increment_state_action_exploration_number(self, s: State, action_id: int, increment: int = 1) -> None:
        self.explored_dict[(s, action_id)] += increment

    def train(self, n_steps: int):
        
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """

        env: Labyrinth = self.env

        for nth_step in range(1, n_steps+1):

            # On reset rapidement l'environnement à chaque step.
            env.reset()
            print(f'[i] Step #{nth_step} - Gamma : {self.gamma}, Epsilon : {self.epsilon}, Alpha : {self.alpha}')

            # Tant que l'exploration n'est pas terminée (sortie), on continue à explorer.
            while not env.is_done():
                
                # On choisit une exploration aléatoire avec une probabilité epsilon
                if random.uniform(0, 1) < self.epsilon:
                    action: int = random.choice(env.get_all_valid_actions())
                else:

                    # Pour le bonus d'exploration, on récupère le nombre de fois ou l'action a été effectuée dans ce state,
                    # puis on le croise avec le paramètre c.
                    action_with_exploration_bonus: list[float] = [qv + ((self.c) / (self.get_state_action_exploration_number(s=env.get_observation(), action_id=aid) + 1))
                                                     for aid, qv in enumerate(self.get_state_value(env.get_observation()))]

                    action: int = np.argmax(action_with_exploration_bonus)

                # Puis on compute le move.
                # (on garde un backup du state précédent pour pouvoir update la bonne qvalue)
                previous_state: State = env.get_observation()
                reward: float = env.step(action=action)
                self.increment_state_action_exploration_number(s=previous_state, action_id=action, increment=1)

                # Ensuite, on vérifie quelle est la meilleure prochaine action
                best_next_action: int = np.argmax(self.get_state_value(env.get_observation()))
                    
                # Enfin, on compute la nouvelle valeur de Q dans la Qtable
                td_target: float = reward + self.gamma * self.get_state_value(s=env.get_observation())[best_next_action]
                td_error: float = (1 - self.alpha) * self.get_state_value(s=previous_state)[action]
                new_qvalue: float = td_error + self.alpha * td_target
                self.set_state_value(s=previous_state, action_id=action, value=new_qvalue)

