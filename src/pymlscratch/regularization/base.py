class BaseRegularization:
    """
        This class implements the Base Regularization
    """

    def __init__(self, *args, **kwargs):
        pass

    def regularize(self, theta):
        return 0

    def gradient(self, theta):
        return 0
