from .l1 import L1
from .l2 import L2
from .elastic_net import ElasticNet
from .base import BaseRegularization
class RegularizationFactory:
    @staticmethod
    def create(reg_type, *args, **kwargs):
        if reg_type == 'l1':
            return L1(*args, **kwargs)
        elif reg_type == 'l2':
            return L2(*args, **kwargs)
        elif reg_type == 'elastic_net':
            return ElasticNet(*args, **kwargs)
        else:
            return BaseRegularization(*args, **kwargs)