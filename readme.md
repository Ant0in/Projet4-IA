# Reinforcement Learning - 2024

## Description

This project implements reinforcement learning algorithms, including Value Iteration and Q-learning, with a focus on $\epsilon$-greedy and aggressive exploration strategies to enhance decision-making in uncertain environments. For more details, please refer to `./pdf/consignes.pdf`.

<p align="center">
  <img src="./rapport/src/intro2.png" alt="intro problem" width="450"/>
  <br/>
  <strong>Instance of the best policy calculated after 10 000 steps for the introductive maze.</strong>
</p>


## Usage

Make sure you have `Python >= 3.11` installed.


### Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Ant0in/Projet4-IA.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Projet4-IA/
   ```

3. Install and setup poetry env:
   
   ```bash
   pip install poetry
   poetry install
   poetry shell
   poetry update // optional
   ```

4. Run the project:

   ```bash
   python ".\main.py" (--options)
   ```

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software.

## Acknowledgements

This project was developed for the Artificial Intelligence course `INFO-F311`. Special thanks to `Tom Lenaerts & Pascal Tribel (ULB)` for their guidance and support.

