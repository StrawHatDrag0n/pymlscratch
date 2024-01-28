from ..tree.cart import Cart

class TreeGrowingAlgoFactory:
    @classmethod
    def create(cls,
               algo,
               impurity,
               min_samples_leaf,
               max_depth,
               calculate_target_value,
               *args,
               **kwargs):
        if algo == 'CART':
            return Cart(impurity,
                        min_samples_leaf,
                        max_depth,
                        calculate_target_value,
                        *args,
                        **kwargs)