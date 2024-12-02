

from .env import Labyrinth
from .agent_container import AgentScoreContainer
from .pickle_helper import PickleHelper

import numpy as np
import random
import time
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


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

        # On garde un objet container pour les reward de l'agent (rapport)
        self.agent_container: AgentScoreContainer = AgentScoreContainer(init_score=0.0)
        

    def get_q_table(self) -> np.ndarray:
        
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """

        return self.qtable

    def set_q_table(self, new_qtable: np.ndarray) -> None:
        self.qtable = new_qtable

    def load_model(self, fp: str) -> None:
        # On load le modèle pour Qlearning (ndarray 3d)
        data: np.ndarray | None = PickleHelper.pickle_safeload(fp=fp)
        if data is not None: self.set_q_table(new_qtable=data)

    def save_model(self, fp: str) -> bool:
        # On save le modèle pour Qlearning (ndarray 3d)
        rc: bool = PickleHelper.pickle_safedump(fp=fp, data=self.qtable)
        return rc

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

    def plot_state_exploration_heatmap(self) -> None:
        
        exploration_table: np.ndarray = np.zeros(self.env.get_map_size())
        for s in self.possible_states:
            v: int = 0
            for a in self.env.get_all_actions(): v += self.explored_dict[(s, a)]
            exploration_table[s[0], s[1]] = v
        
        assert exploration_table.ndim == 2, f"Expected 2D array of shape (height, width), got shape {exploration_table.shape}"
        sns.heatmap(exploration_table, annot=True, cbar_kws={'label': 'Exploration Frequency'})
        plt.show()

    def train(self, n_steps: int, verbose: bool = True) -> None:
        
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """

        env: Labyrinth = self.env
        env.reset()
        start_time: int = time.time()

        for nth_step in range(1, n_steps+1):

            if verbose: print(f'[i] Step #{nth_step} - Gamma : {self.gamma}, Epsilon : {self.epsilon}, Alpha : {self.alpha}, C : {self.c}')

            # Tant que l'exploration n'est pas terminée (sortie), on continue à explorer. Sinon reset.
            if env.is_done():
                self.agent_container.add_score_to_memory_then_reset()  # Stockage du score avant reset
                self.agent_container.add_steps_to_memory_then_reset()  # Stockage des steps avant reset
                init_state: State = env.reset()
                env.set_state(state=init_state)
            
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
            
            self.agent_container.increment_current_score(v=reward)  # On incrémente le score de l'agent
            self.agent_container.increment_current_steps(s=1)  # On incrémente les steps de l'agent (1)

            # Ensuite, on vérifie quelle est la meilleure prochaine action (+bonus)
            best_next_action: int = np.argmax([qv + ((self.c) / (self.get_state_action_exploration_number(s=env.get_observation(), action_id=aid) + 1))
                                                    for aid, qv in enumerate(self.get_state_value(env.get_observation()))])
                
            # Enfin, on compute la nouvelle valeur de Q dans la Qtable
            td_target: float = reward + self.gamma * self.get_state_value(s=env.get_observation())[best_next_action]
            td_error: float = (1 - self.alpha) * self.get_state_value(s=previous_state)[action]
            new_qvalue: float = td_error + self.alpha * td_target
            self.set_state_value(s=previous_state, action_id=action, value=new_qvalue)


        # Infos (verbose)

        if verbose:

            delta_time: float = time.time() - start_time
            completions: int = len(self.agent_container.scores)
            avg_score: float    = 0 if completions == 0 else sum(self.agent_container.scores) / completions
            best_score: float   = 0 if completions == 0 else max(self.agent_container.scores)
            worst_score: float  = 0 if completions == 0 else min(self.agent_container.scores)

            print('\n\n' + '-'*15 + ' Stats ' + '-'*15)
            print(f'[!] Entrainement terminé en {delta_time:.2f}s')
            print(f'[!] Total steps : {n_steps}')
            print(f'[!] Maze Completions : {completions}')
            print(f'[!] Best score : {best_score}')
            print(f'[!] Worst score : {worst_score}')
            print(f'[!] Score average : {avg_score:.2f}')
            print('-'*37 + '\n\n')

