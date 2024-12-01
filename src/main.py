
from rl.env import Labyrinth
from rl.qlearning import QLearning
from rl.value_iteration import ValueIteration
from rl.agent_container import AgentScoreContainer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray



def plot_values(values: NDArray[np.float64]) -> None:
    """
    Plots a heatmap representing the state values in a grid world.

    Parameters:
    - values (NDArray[np.float64]): A 2D numpy array of shape (height, width) where each element 
                                    represents the computed value of that state.

    Returns:
    - None: Displays the plot.
    """
    assert values.ndim == 2, f"Expected 2D array of shape (height, width), got shape {values.shape}"
    sns.heatmap(values, annot=True, cbar_kws={'label': 'Value'})
    plt.show()

def plot_qvalues(q_values: NDArray[np.float64], action_symbols: list[str]) -> None:
    """
    Plots a heatmap of the maximum Q-values in each state of a grid world and overlays symbols
    to represent the optimal action in each state.

    Parameters:
    - q_values (NDArray[np.float64]): A 3D numpy array of shape (height, width, n_actions), where each cell contains Q-values
                                      for four possible actions (up, down, right, left).
    - env (Labyrinth): The environment instance to access action symbols.

    Returns:
    - None: Displays the plot.
    """
    assert q_values.ndim == 3, f"Expected 3D array of shape (height, width, n_actions), got shape {q_values.shape}"
    assert q_values.shape[-1] == len(action_symbols), f"Number of action symbols should match the number of actions"
    height, width = q_values.shape[:2]


    # Calculate the best action and max Q-value for each cell
    best_actions = np.argmax(q_values, axis=2)
    max_q_values = np.max(q_values, axis=2)

    # Plotting the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(max_q_values, origin="upper")
    plt.colorbar(label="Max Q-value")
    # Overlay best action symbols
    for i in range(height):
        for j in range(width):
            action_symbol = action_symbols[best_actions[i, j]]
            plt.text(j, i, action_symbol, ha='center', va='center', color='black', fontsize=12)

    # Labels and layout
    plt.title("Q-value Heatmap with Optimal Actions")
    plt.grid(False)
    plt.show()

def random_moves(env: Labyrinth, n_steps: int) -> None:
    """
    Makes random moves in the environment and renders each step.

    Parameters:
    - env (Labyrinth): The environment instance where random moves will be performed.
    - n_steps (int): Number of random steps to perform.

    Returns:
    - None
    """
    env.reset()
    env.render()
    episode_rewards = 0
    for s in range(n_steps):

        random_action = np.random.choice(env.get_all_actions())
        reward = env.step(random_action)
        done = env.is_done()
        episode_rewards += reward
        if done:
            print("collected reward =", episode_rewards)
            env.reset()
            episode_rewards = 0
        #env.render()



if __name__ == "__main__":
    
    env = Labyrinth(malfunction_probability=0.1)
    env.reset()

    # Uncomment for random moves
    # random_moves(env, 10_000)

    # algo: ValueIteration = ValueIteration(env=env, gamma=0.9)
    # algo.train(40)
    # plot_values(algo.get_value_table())

    # Uncomment for Q-learning

    steps: int = 10_000

    algo: QLearning = QLearning(env=env, gamma=1, alpha=.1, epsilon=0, c=100)
    algo.train(steps)
    print(algo.agent_container.get_avg_reward_vector())
    # plot_qvalues(algo.get_q_table(), action_symbols=Labyrinth.ACTION_SYMBOLS)

