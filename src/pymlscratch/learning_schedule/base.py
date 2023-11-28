class BaseLearningSchedule:
    def __init__(self, *args, **kwargs):
        pass

    def get_learning_rate(self, t):
        return t