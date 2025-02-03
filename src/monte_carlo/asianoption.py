from src.monte_carlo.baseoption import BaseOption
import numpy as np
from scipy.stats import norm

class AsianOption(BaseOption):
    def __init__(self, S0, K, T, r, sigma, n_steps, option_type='arithmetic'):
        """
        Initialize Asian option

        :param n_steps: Number of time steps for averaging
        :param option_type: 'arithmetic' or 'geometric'
        """
        super().__init__(S0, K, T, r, sigma)
        self.n_steps = n_steps
        self.option_type = option_type
        
    def payoff(self, stock_paths) -> np.ndarray:
        """
        Calculate Asian option payoffs based on 'arithmetic' or 'geometric' averaging

        :param stock_paths: Stock price paths
        :return: Option payoffs
        """
        if self.option_type == 'arithmetic':
            average = np.mean(stock_paths, axis=1)
        else:  # geometric
            average = np.exp(np.mean(np.log(stock_paths), axis=1))
        
        return np.maximum(average - self.K, 0)
    
    def geometric_price(self) -> float:
        """
        Analytical price for geometric Asian option (used as control variate)

        :return: Option price
        """
        dt = self.T / self.n_steps
        adjusted_sigma = self.sigma * np.sqrt((2 * self.n_steps + 1) / (6 * (self.n_steps + 1)))
        adjusted_r = (self.r - 0.5 * self.sigma**2) * (self.n_steps + 1) / (2 * self.n_steps) + \
                    0.5 * adjusted_sigma**2
        
        d1 = (np.log(self.S0 / self.K) + (adjusted_r + 0.5 * adjusted_sigma**2) * self.T) / \
             (adjusted_sigma * np.sqrt(self.T))
        d2 = d1 - adjusted_sigma * np.sqrt(self.T)
        
        return self.S0 * np.exp((adjusted_r - self.r) * self.T) * norm.cdf(d1) - \
               self.K * np.exp(-self.r * self.T) * norm.cdf(d2)