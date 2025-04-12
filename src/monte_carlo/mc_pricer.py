from src.constructors.asianoption import AsianOption
import numpy as np


class MCPricer:
    def __init__(self, n_sims=10000, n_steps=252):
        """
        Initialize Monte Carlo pricer
        
        :param n_sims: Number of simulation paths
        :param n_steps: Number of time steps
        """
        self.n_sims = n_sims
        self.n_steps = n_steps
    
    def generate_paths(self, option) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion
        
        :param option: Option object
        :return: Stock price paths
        """
        dt = option.T / self.n_steps
        nudt = (option.r - 0.5 * option.sigma**2) * dt
        sigmasqrtdt = option.sigma * np.sqrt(dt)
        
        Z = np.random.normal(0, 1, (self.n_sims, self.n_steps))
        increments = nudt + sigmasqrtdt * Z
        
        logpaths = np.log(option.S0) + np.cumsum(increments, axis=1)
        paths = np.exp(logpaths)
        
        return paths
    
    def calculate_payoffs(self, option, use_control_variate=False):
        """
        Helper method to calculate payoffs with or without control variate
        
        :param option: Option object
        :param use_control_variate: True to use control variate technique
        :return: Option payoffs
        """
        paths = self.generate_paths(option)

        if use_control_variate and isinstance(option, AsianOption) and option.option_type == 'arithmetic':
            arithmetic_payoffs = option.payoff(paths)

            geometric_option = AsianOption(
                option.S0, option.K, option.T, option.r, option.sigma,
                option.n_steps, 'geometric'
            )
            geometric_payoffs = geometric_option.payoff(paths)
            geometric_price = geometric_option.geometric_price()

            cov_matrix = np.cov(arithmetic_payoffs, geometric_payoffs)
            beta = -cov_matrix[0, 1] / np.var(geometric_payoffs)

            return arithmetic_payoffs + beta * (geometric_payoffs - geometric_price)
        else:
            return option.payoff(paths)

    def price(self, option, use_control_variate=False) -> float:
        """
        Calculate option price
        
        :param option: Option object
        :param use_control_variate: True to use control variate technique
        :return: Option price
        """
        payoffs = self.calculate_payoffs(option, use_control_variate)
        return np.exp(-option.r * option.T) * np.mean(payoffs)
    
    def std_error(self, option, use_control_variate=False, n_bootstrap=1000) -> float:
        """
        Calculate standard error of the price estimate using bootstrapping
        
        :param option: Option object
        :param use_control_variate: True to use control variate technique
        :param n_bootstrap: Number of bootstrap samples to generate (default: 1000)
        :return: Standard error of the price estimate
        """
        payoffs = self.calculate_payoffs(option, use_control_variate)
        discount_factor = np.exp(-option.r * option.T)
        
        # Generate bootstrap samples
        bootstrap_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(payoffs, size=len(payoffs), replace=True)
            bootstrap_means[i] = discount_factor * np.mean(bootstrap_sample)
        
        # Standard error is the standard deviation of the bootstrap distribution
        return np.std(bootstrap_means)