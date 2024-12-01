

import seaborn as sns
import matplotlib.pyplot as plt


class AgentScoreContainer:

    def __init__(self, init_score: float = 0.0) -> None:
        
        self._init_score: float = init_score
        self._scores: list[float] = list()
        self._current_score: float = init_score
        self._steps_vector: list[int] = list()
        self._current_steps: int = 0

    @property
    def scores(self) -> list[float]:
        return self._scores
    
    @property
    def init_score(self) -> float:
        return self._init_score

    @property
    def current_score(self) -> float:
        return self._current_score

    @property
    def steps_vector(self) -> list[int]:
        return self._steps_vector
    
    @property
    def current_steps(self) -> int:
        return self._current_steps

    def set_current_steps(self, s: int) -> None:
        self._current_steps = s

    def increment_current_steps(self, s: int) -> None:
        self.set_current_steps(s=(self.current_steps + s))

    def reset_current_steps(self) -> None:
        self.set_current_steps(s=0)

    def add_steps_to_memory_then_reset(self) -> None:
        self.steps_vector.append(self.current_steps)
        self.reset_current_steps()

    def set_current_score(self, v: float) -> None:
        self._current_score = v

    def increment_current_score(self, v: float) -> None:
        self.set_current_score(v=(self.current_score + v))

    def reset_current_score(self) -> None:
        self.set_current_score(v=self.init_score)

    def add_score_to_memory_then_reset(self) -> None:
        self.scores.append(self.current_score)
        self.reset_current_score()

    def get_avg_reward_vector(self) -> list[float]:
        avg_reward_vec: list[float] = [score / step for step, score in zip(self.steps_vector, self.scores)]
        return avg_reward_vec

    @staticmethod
    def round_to_highest_digit(num):
        from math import log10
        if num == 0: return 0
        power = int(log10(abs(num)))
        factor = 10**power
        return round(num / factor) * factor
    
    def plot_avg_reward(self, steps: int) -> None:

        avg_reward_vec: list[float] = self.get_avg_reward_vector()
        
        sns.barplot(x=range(len(avg_reward_vec)), y=avg_reward_vec, color='skyblue')
        plt.title(f"Average reward par completion ({steps} steps)")
        plt.xlabel("Completions")
        plt.ylabel("Reward Average")
        xticks: range = range(0, len(avg_reward_vec), self.round_to_highest_digit(steps))
        plt.xticks(xticks)
        plt.show()

    def plot_scores(self, steps: int) -> None:
        sns.barplot(x=range(len(self.scores)), y=self.scores, color='skyblue')
        plt.title(f"Score obtenus pour chaque completion du labyrinthe ({steps} steps)")
        plt.xlabel("Completion")
        plt.ylabel("Scores")
        xticks: range = range(0, len(self.scores), self.round_to_highest_digit(steps))
        plt.xticks(xticks)
        plt.show()

    

