

from src.rl import QLearning, ValueIteration, Labyrinth
from src.main import plot_qvalues, plot_values
import argparse



class MainExecutor:

    @staticmethod
    def main_parser() -> dict:

        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        parser.add_argument("p", type=float, help="Facteur déterministe (entre 0 et 1).")
        parser.add_argument("steps", type=int, help="Nombre de steps d'entraînement (int).")
        parser.add_argument("algo", choices=["valueiteration", "qlearning"], help="Algorithme d'entraînement : 'valueiteration' / 'qlearning'.")
        
        # Paramètres
        parser.add_argument("--gamma", type=float, required=False, help="Facteur de discount gamma (float). (Obligatoire)")
        parser.add_argument("--alpha", type=float, required=False, help="Taux d'apprentissage alpha (float). (Obligatoire pour Q-Learning)")
        parser.add_argument("--epsilon", type=float, required=False, help="Paramètre epsilon pour l'exploration (float). (Obligatoire pour Q-Learning)")
        parser.add_argument("--c", type=float, required=False, help="Bonus d'exploration c (float). (Obligatoire pour Q-Learning)")

        # Paramètres optionnels
        parser.add_argument("--verbose", action="store_true", default=True, help="Flag pour activer le mode verbose (par défaut activé).")
        parser.add_argument("--save_model", type=str, help="Path du fichier pour sauvegarder le modèle (string).")
        parser.add_argument("--load_model", type=str, help="Path du fichier pour charger le modèle (string).")

        args: argparse.Namespace = parser.parse_args()

        if args.algo == "qlearning":
            if args.alpha is None or args.epsilon is None or args.c is None or args.gamma is None:
                parser.error("Pour l'algorithme 'qlearning', les paramètres '--gamma', '--alpha', '--epsilon' et '--c' sont obligatoires.")

        if args.algo == "valueiteration":
            if args.gamma is None:
                parser.error("Pour l'algorithme 'valueiteration', le paramètre '--gamma' est obligatoire.")

        return vars(args)

    @staticmethod
    def execute_valueiteration(parameters: dict) -> None:
        
        env: Labyrinth = Labyrinth(malfunction_probability=parameters['p'])
        algo: ValueIteration = ValueIteration(env=env, gamma=parameters['gamma'])

        # Safe load
        if parameters['load_model']:
            algo.load_model(fp=parameters['load_model'])

        # Train
        algo.train(n_updates=parameters['steps'], verbose=parameters['verbose'])
        
        # Safe dump
        if parameters['save_model']:
            algo.save_model(fp=parameters['save_model'])

        plot_values(values=algo.get_value_table())

    @staticmethod
    def execute_qlearning(parameters: dict) -> None:
        
        env: Labyrinth = Labyrinth(malfunction_probability=parameters['p'])
        algo: QLearning = QLearning(
            env=env,
            gamma=parameters['gamma'],
            alpha=parameters['alpha'],
            epsilon=parameters['epsilon'],
            c=parameters['c']
        )

        # Safe load
        if parameters['load_model']:
            algo.load_model(fp=parameters['load_model'])

        # Train
        algo.train(n_steps=parameters['steps'], verbose=parameters['verbose'])
       
        # Safe dump
        if parameters['save_model']:
            algo.save_model(fp=parameters['save_model'])

        plot_qvalues(q_values=algo.get_q_table(), action_symbols=algo.env.ACTION_SYMBOLS)

    @staticmethod
    def main_exec(parameters: dict) -> None:

        match parameters['algo']:
            case 'qlearning': MainExecutor.execute_qlearning(parameters=parameters)
            case 'valueiteration': MainExecutor.execute_valueiteration(parameters=parameters)
            case _: raise NotImplementedError()



# if __name__ == "__maison__" ? >__<"

if __name__ == "__main__":
    
    parameters: dict = MainExecutor.main_parser()
    MainExecutor.main_exec(parameters=parameters)


    # usage example:

    # python .\main.py 0.1 1000 valueiteration --gamma 0.9
    # python .\main.py 0.1 1000 qlearning --gamma 0.9 --alpha 0.1 --epsilon 0.0 --c 100

    # option : 
    # --save_model './model_xyz.pkl'
    # --load_model './model_xyz.pkl'
    # --verbose False | True

