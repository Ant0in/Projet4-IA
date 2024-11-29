

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
        self.possible_states: list[tuple[int, int]] = env.get_valid_states()

        self.value_table: np.ndarray = np.zeros(env.get_map_size())

    def train(self, n_updates: int) -> None:

        env: Labyrinth = self.env

        # On fait n itérations de l'algorithme.
        for nth_update in range(n_updates):

            print(f'[i] Itération #{nth_update}')

            for s in self.possible_states:

                # On récupère les actions possibles, puis on regarde ce qu'elles font en reward et en state terminal
                # On calcule tout les états accessibles ainsi que leurs reward.
                s_prime_reward_dict: dict[tuple[int, int]: float] = dict()

                for a in env.get_all_actions():
                    s_next, r = env.step_without_corruption(action=a, state=s)
                    if s_next not in s_prime_reward_dict:
                        s_prime_reward_dict[s_next] = r

                # Ensuite on cherche à récupérer V_k+1.
                s_prime: list = list(s_prime_reward_dict.keys())
                values: list = []

                for idx, s_p in enumerate(s_prime):
                    
                    cumulative_value: float = 0.0

                    # On récupère les states dus à une malfonctions (proba p/#{s'}-1) puis on ajoute le déplacement prévu (1-p)
                    transition_model: dict = {cs: (env.malfunction_probability / (len(s_prime) - 1)) for cs in s_prime[:idx] + s_prime[idx+1:]}
                    transition_model[s_p] = 1 - env.malfunction_probability

                    for f in transition_model:
                        # On utilise T(s, a, s') * [R(s, a, s') + gamma * Vk(s')]
                        it_v: float = transition_model[f] * (s_prime_reward_dict[f] + (self.gamma * self.get_state_value(s=f)))
                        cumulative_value += it_v

                    values.append(cumulative_value)

                new_v_value: float = max(values)
                self.set_state_value(s=s, value=new_v_value)

    def get_state_value(self, s: tuple[int, int]) -> float:
        return self.get_value_table()[s[0], s[1]]
    
    def set_state_value(self, s: tuple[int, int], value: float) -> None:
        self.get_value_table()[s[0], s[1]] = value

    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        return self.value_table
    

