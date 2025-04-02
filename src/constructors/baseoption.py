from abc import ABC, abstractmethod


class BaseOption(ABC):
    def __init__(self, S0, K, T, r, sigma):
        """
        Initialize base option parameters
        
        :param S0: Initial stock price
        :param K: Strike price
        :param T: Time to maturity in years
        :param r: Risk-free rate
        :param sigma: Volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    @abstractmethod
    def payoff(self, stock_paths):
        """Calculate option payoff given stock price path(s)"""
        pass