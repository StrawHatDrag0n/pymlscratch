from .mse import MSE
from .log_loss import LogLoss
from .hinge import HingeLoss

class LossFactory:
    """
        This class is a factory for cost functions
    """
    @classmethod
    def create(cls, loss):
        if loss == 'mse' or loss == 'mean_squared_error':
            return MSE
        elif loss == 'log_loss' or loss == 'logistic_loss':
            return LogLoss()
        elif loss == 'hinge_loss':
            return HingeLoss()
        else:
            raise ValueError('Loss function {} not found'.format(loss))
