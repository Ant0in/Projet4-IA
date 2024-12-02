

import numpy as np
import time
from .env import Labyrinth
from .pickle_helper import PickleHelper
from .agent_container import AgentScoreContainer

# alias
State = tuple[int, int]



class ValueIteration:

    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    ...
    """
    
    def __init__(self, env: Labyrinth, gamma: float = 1.0) -> None:

        self.env: Labyrinth = env
        self.gamma: float = gamma
        self.possible_states: list[State] = env.get_valid_states()
        self.value_table: np.ndarray = np.zeros(env.get_map_size())

    def load_model(self, fp: str) -> None:
        # On load le modèle pour ValueIteration (ndarray 2d)
        data: np.ndarray | None = PickleHelper.pickle_safeload(fp=fp)
        if data is not None: self.set_value_table(vtable=data)

    def save_model(self, fp: str) -> bool:
        # On save le modèle pour ValueIteration (ndarray 2d)
        rc: bool = PickleHelper.pickle_safedump(fp=fp, data=self.value_table)
        return rc

    def get_value_table(self) -> np.ndarray:
        
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """

        return self.value_table

    def set_value_table(self, vtable: np.ndarray) -> None:
        self.value_table = vtable

    def get_state_value(self, s: State) -> float:
        # Get state value in the value table
        return self.get_value_table()[s[0], s[1]]
    
    def set_state_value(self, s: State, value: float) -> None:
        # Set state value in the value table
        self.get_value_table()[s[0], s[1]] = value

    def train(self, n_updates: int, verbose: bool = True) -> None:

        env: Labyrinth = self.env
        start_time: float = time.time()

        # On fait n itérations de l'algorithme.
        for nth_update in range(1, n_updates+1):

            if verbose: print(f'[i] Itération #{nth_update} - Gamma : {self.gamma}')

            for s in self.possible_states:

                # On récupère les actions possibles, puis on regarde ce qu'elles font en reward et en state terminal
                # On calcule tout les états accessibles ainsi que leurs reward.
                s_prime_reward_dict: dict[State: float] = dict()

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
        
        if verbose:
            
            delta_time: float = time.time() - start_time
            print('\n\n' + '-'*15 + ' Stats ' + '-'*15)
            print(f'[!] Entrainement terminé en {delta_time:.2f}s')
            print(f'[!] Total steps : {n_updates}')
            print('-'*37 + '\n\n')

