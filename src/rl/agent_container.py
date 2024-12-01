

import seaborn as sns
import matplotlib.pyplot as plt


class AgentScoreContainer:

    def __init__(self, init_score: float = 0.0) -> None:
        
        self._init_score: float = init_score
        self._scores: list[float] = list()
        self._current_score: float = init_score

    @property
    def scores(self) -> list[float]:
        return self._scores
    
    @property
    def init_score(self) -> float:
        return self._init_score

    @property
    def current_score(self) -> float:
        return self._current_score
    
    def set_current_score(self, v: float) -> None:
        self._current_score = v

    def increment_current_score(self, v: float) -> None:
        self.set_current_score(v=(self.current_score + v))

    def reset_current_score(self) -> None:
        self.set_current_score(v=self.init_score)

    def add_score_to_memory_then_reset(self) -> None:
        self.scores.append(self.current_score)
        self.reset_current_score()

    def plot_scores(self, steps: int) -> None:

        sns.barplot(x=range(len(self.scores)), y=self.scores, color='skyblue')
        plt.title(f"Score obtenus pour chaque completion du labyrinthe ({steps} steps)")
        plt.xlabel("Complétions")
        plt.ylabel("Scores")
        xticks = range(0, len(self.scores), 1000)
        plt.xticks(xticks)
        plt.show()

