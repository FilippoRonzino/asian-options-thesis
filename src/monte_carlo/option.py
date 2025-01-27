import numpy as np
from abc import ABC, abstractmethod

class Option(ABC):
    def __init__(self, S0, K, T, r, sigma):
        """
        Base Option Class
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    @abstractmethod
    def payoff(self, ST):
        """
        Abstract method to calculate option payoff
        
        Parameters:
        ST (float or np.array): Stock price at maturity
        """
        pass
    
    @abstractmethod
    def price(self):
        """
        Abstract method to calculate option price
        """
        pass