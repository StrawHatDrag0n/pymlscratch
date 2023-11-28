from .base import BaseLearningSchedule

class SimpleLearningSchedule(BaseLearningSchedule):
    def __init__(self, t0=5, t1=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = t0
        self.t1 = t1

    def get_learning_rate(self, t):
        return self.t0 / (t + self.t1)