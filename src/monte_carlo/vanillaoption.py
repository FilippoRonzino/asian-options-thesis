from src.monte_carlo.baseoption import BaseOption
import numpy as np

class VanillaOption(BaseOption):
    def __init__(self, S0, K, T, r, sigma, option_type='european'):
        """
        Initialize vanilla option
        
        :param option_type: 'european' or 'american'
        """
        super().__init__(S0, K, T, r, sigma)
        self.option_type = option_type
    
    def payoff(self, stock_paths) -> np.ndarray:
        if self.option_type == 'european':
            return np.maximum(stock_paths[:, -1] - self.K, 0)
        else:  # american
            exercise_values = np.maximum(stock_paths - self.K, 0)
            return np.max(exercise_values, axis=1)
