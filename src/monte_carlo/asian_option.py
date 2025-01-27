# asian_option.py
import numpy as np
from scipy.stats import norm

class GeometricAsianOption:
    def __init__(self, S0, K, T, r, sigma, n_steps):
        """
        Geometric Asian Option Class
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        n_steps (int): Number of time steps for averaging
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        
    def payoff(self, ST):
        """Calculate the payoff for geometric Asian option"""
        return np.maximum(ST - self.K, 0)
    
    def price(self):
        """
        Calculate geometric Asian option price using analytical formula
        """
        dt = self.T / self.n_steps
        
        # Adjusted parameters for geometric Asian option
        sigma_adj = self.sigma * np.sqrt((self.n_steps + 2) / (6 * (self.n_steps + 1)))
        r_adj = (self.r - 0.5 * self.sigma**2) * (self.n_steps + 1) / (2 * self.n_steps) + \
                self.r / 2
        
        # Black-Scholes formula with adjusted parameters
        d1 = (np.log(self.S0 / self.K) + (r_adj + 0.5 * sigma_adj**2) * self.T) / \
             (sigma_adj * np.sqrt(self.T))
        d2 = d1 - sigma_adj * np.sqrt(self.T)
        
        price = np.exp(-self.r * self.T) * \
                (self.S0 * np.exp(r_adj * self.T) * norm.cdf(d1) - \
                 self.K * norm.cdf(d2))
        
        return price

class ArithmeticAsianOption:
    def __init__(self, S0, K, T, r, sigma, n_steps):
        """
        Arithmetic Asian Option Class
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        n_steps (int): Number of time steps for averaging
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        
    def payoff(self, price_path):
        """
        Calculate the payoff for arithmetic Asian option
        
        Parameters:
        price_path (np.array): Array of stock prices over time
        """
        arithmetic_mean = np.mean(price_path, axis=0)
        return np.maximum(arithmetic_mean - self.K, 0)