from pymlscratch.impurity.gini import GiniImpurity
from pymlscratch.impurity.mean_absolute_error import MAE

class ImpurityFactory:
    @classmethod
    def create(cls, impurity):
        if impurity == 'gini':
            return GiniImpurity()
        if impurity == 'MAE':
            return MAE()
