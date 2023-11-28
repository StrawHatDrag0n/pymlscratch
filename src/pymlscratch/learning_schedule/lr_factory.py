from .base import BaseLearningSchedule
from .simple import SimpleLearningSchedule


class LearningScheduleFactory:
    @staticmethod
    def create(lr_type='constant', *args, **kwargs):
        if lr_type == 'simple':
            return SimpleLearningSchedule(*args, **kwargs)
        else:
            return BaseLearningSchedule(*args, **kwargs)