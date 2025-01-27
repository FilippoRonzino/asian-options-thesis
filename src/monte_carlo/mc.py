# monte_carlo.py
import numpy as np
from monte_carlo.asian_option import GeometricAsianOption

def generate_paths(S0, T, r, sigma, n_steps, n_sims):
    """
    Generate stock price paths using geometric Brownian motion
    
    Parameters:
    S0 (float): Initial stock price
    T (float): Time to maturity
    r (float): Risk-free rate
    sigma (float): Volatility
    n_steps (int): Number of time steps
    n_sims (int): Number of simulation paths
    """
    dt = T / n_steps
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_steps, n_sims))
    paths = np.zeros((n_steps + 1, n_sims))
    paths[0] = S0
    
    for t in range(1, n_steps + 1):
        paths[t] = paths[t-1] * np.exp(nudt + sidt * Z[t-1])
    
    return paths

def price_asian_mc(option, n_sims, control_variate=False):
    """
    Price Asian option using Monte Carlo simulation
    
    Parameters:
    option (ArithmeticAsianOption): Option to price
    n_sims (int): Number of simulation paths
    control_variate (bool): Whether to use control variate technique
    """
    paths = generate_paths(option.S0, option.T, option.r, option.sigma, 
                         option.n_steps, n_sims)
    
    # Calculate arithmetic Asian option price
    arithmetic_payoffs = option.payoff(paths)
    arithmetic_price = np.exp(-option.r * option.T) * np.mean(arithmetic_payoffs)
    
    if control_variate:
        # Create geometric Asian option with same parameters
        geometric = GeometricAsianOption(option.S0, option.K, option.T, 
                                       option.r, option.sigma, option.n_steps)
        geometric_analytical = geometric.price()
        
        # Calculate geometric average for paths
        geometric_mean = np.exp(np.mean(np.log(paths), axis=0))
        geometric_payoffs = np.maximum(geometric_mean - option.K, 0)
        geometric_mc = np.exp(-option.r * option.T) * np.mean(geometric_payoffs)
        
        # Calculate optimal control variate coefficient
        covariance = np.cov(arithmetic_payoffs, geometric_payoffs)[0,1]
        variance = np.var(geometric_payoffs)
        theta = covariance / variance
        
        # Apply control variate adjustment
        arithmetic_price += theta * (geometric_analytical - geometric_mc)
    
    return arithmetic_price