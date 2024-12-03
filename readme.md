# Reinforcement Learning - 2024

## Description

This project implements reinforcement learning algorithms, including Value Iteration and Q-learning, with a focus on $\epsilon$-greedy and aggressive exploration strategies to enhance decision-making in uncertain environments. For more details, please refer to `./pdf/consignes.pdf`.

<p align="center">
  <img src="./rapport/src/intro2.png" alt="intro problem" width="450"/>
  <br/>
  <strong>Instance of the best policy calculated after 10 000 steps for the introductive maze.</strong>
</p>


## Usage

Make sure you have `3.10 <= Python < 3.13` installed.


## Features

- **Q-Learning and Value Iteration Implementations**: Algorithms designed to help the agent learn the optimal policy for navigating a maze.

- **Model Persistence**: Load and save trained models using `pickle`, allowing the agent to retain its training for future use.

- **Highly Configurable Parameters**: Easily adjust algorithm parameters to fine-tune performance. Refer to the `Usage` section for detailed instructions. 

## Install the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Ant0in/Projet4-IA.git
   ```

2. Navigate to the **project directory**:
   ```bash
   cd Projet4-IA/
   ```

3. Install **Poetry** (if not already installed):
   ```bash
   pip install poetry
   ```

4. **Install** the project dependencies using **Poetry**:
   ```bash
   poetry install
   ```

5. Activate the **virtual environment**:
   ```bash
   poetry shell
   ```

6. **Update** dependencies to their latest compatible versions *(Optional)*:
   ```bash
   poetry update
   ```

## Usage

When installed, run the project through the `main.py` file. 
The main file is located at `Projet4-IA/main.py`.

### Required Parameters (Non-positional):
- The probability `p` of the environment not being deterministic. Value must be a **float** between `0` and `1`.
- The number of `steps` for training. Value must be a **positive integer** or `0`.
- The algorithm used to train the agent. Possible choices are `valueiteration` or `qlearning`.

For the 2 possible algorithm, specific training parameters are required **(but positional)**. See below :

- **Value Iteration** Algorithm `valueiteration` :
   - **Discount factor** `gamma`, which decreases the positive reward gained by the agent over time. Value must be a **float** between `0` and `1` (where `1` means **no discount**).

- **Qlearning** Algorithm `qlearning` :
   - **Discount factor** `gamma` as described above.
   - **Learning rate** `alpha`, which controls how much the agent learns from new training steps compared to trusting its previous knowledge. Value must be a **float** between `0` and `1`.
   - **Greed factor** `epsilon`, which indicates the probability of the agent taking a random action instead of the best-known action. Value must be a **float** between `0` and `1`.
   - **Exploration bonus** `c`, which increases the agent's interest in states it has never explored before. Value must be a **positive integer** or `0`.

### Optional parameters : 

- `--verbose`: Enables or disables verbose mode. **Enabled by default**.
- `--save_model "filepath"`: Saves training data to a `.pkl` file. Saving happens **after training**.
- `--load_model "filepath"`: Loads training data from a `.pkl` file. Loading happens **before training**.


### Running Examples:

1. Running the Value Iteration algorithm with a high `gamma` for **10,000 steps** in a `0.1` probability undeterministic environment, and saving training data:

   ```bash
   python .\main.py 0.1 10000 valueiteration --gamma 0.9 --save_model './model_xyz.pkl'
   ```

2. Loading a model and training it for 5000 steps with the Q-Learning algorithm, using a high **exploration bonus** and very low **exploitation of knowledge (`epsilon`)**:

   ```bash
   python .\main.py 0.1 5000 qlearning --gamma 0.9 --alpha 0.1 --epsilon 0.0 --c 1000 --load_model './model_xyz.pkl'
   ```


## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software.

## Acknowledgements

This project was developed for the Artificial Intelligence course `INFO-F311`. Special thanks to `Tom Lenaerts & Pascal Tribel (ULB)` for their guidance and support.

