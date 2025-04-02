import matplotlib.pyplot as plt
from constructors.asianoption import AsianOption
from monte_carlo.mc_pricer import MCPricer


def test_control_variate(S0, K, T, r, sigma, n_steps, n_sims_list, plot=True):
    std_errors_no_cv = []
    std_errors_cv = []
    variance_reduction_ratios = []

    for n_sims in n_sims_list:
        option = AsianOption(S0, K, T, r, sigma, n_steps, option_type='arithmetic')
        pricer = MCPricer(n_sims=n_sims, n_steps=n_steps)

        std_error_no_cv = pricer.std_error(option, use_control_variate=False)
        std_error_cv = pricer.std_error(option, use_control_variate=True)

        std_errors_no_cv.append(std_error_no_cv)
        std_errors_cv.append(std_error_cv)

        variance_reduction_ratio = (std_error_no_cv ** 2) / (std_error_cv ** 2)
        variance_reduction_ratios.append(variance_reduction_ratio)
    
    if plot:
        plot_standard_errors(n_sims_list, std_errors_no_cv, std_errors_cv)
        plot_variance_reduction(n_sims_list, variance_reduction_ratios)
    
    return n_sims_list, std_errors_no_cv, std_errors_cv, variance_reduction_ratios

def plot_standard_errors(n_sims_list, std_errors_no_cv, std_errors_cv):
    plt.figure(figsize=(12, 6))
    plt.plot(n_sims_list, std_errors_no_cv, label='Without Control Variate', marker='o')
    plt.plot(n_sims_list, std_errors_cv, label='With Control Variate', marker='x')
    plt.title('Standard Errors of Asian Option Pricing')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Standard Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_variance_reduction(n_sims_list, variance_reduction_ratios):
    plt.figure(figsize=(12, 6))
    plt.plot(n_sims_list, variance_reduction_ratios, label='Variance Reduction Ratio', marker='s')
    plt.title('Variance Reduction Ratio (Without CV / With CV)')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Variance Reduction Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_real_data(df, n_steps, n_sims_list, plot=True):
    row = df.sample(n=1).iloc[0]  # Select a random row
    
    S0 = row['UNDERLYING_PRICE']
    K = row['STRIKE_PX']
    T = row['TIME_TO_MATURITY']
    r = row['RISK_FREE_RATE']
    sigma = row['VOLATILITY_30D']/100  # Convert percentage to decimal

    std_errors_no_cv = []
    std_errors_cv = []
    variance_reduction_ratios = []

    for n_sims in n_sims_list:
        option = AsianOption(S0, K, T, r, sigma, n_steps, option_type='arithmetic')
        pricer = MCPricer(n_sims=n_sims, n_steps=n_steps)

        std_error_no_cv = pricer.std_error(option, use_control_variate=False)
        std_error_cv = pricer.std_error(option, use_control_variate=True)

        std_errors_no_cv.append(std_error_no_cv)
        std_errors_cv.append(std_error_cv)

        variance_reduction_ratio = (std_error_no_cv ** 2) / (std_error_cv ** 2)
        variance_reduction_ratios.append(variance_reduction_ratio)
    
    print(f"Selected Option: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Standard Errors Without Control Variate: {std_errors_no_cv}")
    print(f"Standard Errors With Control Variate: {std_errors_cv}")
    print(f"Variance Reduction Ratios: {variance_reduction_ratios}")
    
    if plot:
        plot_standard_errors(n_sims_list, std_errors_no_cv, std_errors_cv)
        plot_variance_reduction(n_sims_list, variance_reduction_ratios)
    
    return n_sims_list, std_errors_no_cv, std_errors_cv, variance_reduction_ratios
