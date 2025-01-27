import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

def simulate_option_prices_random(n_sim=1000, N=1000, sigma=2, r=0.5, K=1.5, save_dir="plots"):
    """
    Simulate and compare Asian and European option prices, displaying a random path and average prices side-by-side.
    
    Parameters:
    n_sim: number of simulations
    N: number of time steps
    sigma: volatility
    r: risk-free rate
    K: strike price
    save_dir: directory to save the final plot
    """
    # Enable LaTeX rendering
    rc("text", usetex=True)
    rc("font", family="serif", size=12)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Time parameters
    dt = 1.0 / N
    t = np.linspace(0, 1, N + 1)

    # Arrays to store results
    european_payoffs = []
    asian_payoffs = []
    all_paths = []
    all_averages = []

    for j in range(n_sim):
        # Generate random path using geometric Brownian motion
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.cumsum(dW)
        S = np.exp(sigma * W + (r - sigma**2 / 2) * t[1:])
        S = np.insert(S, 0, 1)  # Add initial price S0 = 1

        # Calculate averages and payoffs
        A = np.mean(S)
        european_payoff = max(S[-1] - K, 0)
        asian_payoff = max(A - K, 0)

        european_payoffs.append(european_payoff)
        asian_payoffs.append(asian_payoffs)
        all_paths.append(S)
        all_averages.append(A)

    # Select a random path and its average
    random_index = np.random.randint(0, n_sim)
    random_path = all_paths[random_index]
    random_avg_price = all_averages[random_index]

    # Calculate final average prices
    avg_european = np.mean(european_payoffs)
    avg_asian = np.mean(asian_payoffs)

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Plot the random path on the left
    color = 'darkred' if random_path[-1] > random_avg_price else 'darkgreen'
    axs[0].plot(t, random_path, color=color, linewidth=2, label='Price Path')
    axs[0].axhline(y=K, color='red', linestyle='-', linewidth=2, label='Strike Price')
    axs[0].axhline(y=random_avg_price, color='darkgreen', linestyle='--', linewidth=2, label='Average Price')

    axs[0].set_title(r'\textbf{Random Simulated Stock Price Path}', fontsize=16)
    axs[0].set_xlabel(r'\textbf{Time}', fontsize=14)
    axs[0].set_ylabel(r'\textbf{Price}', fontsize=14)

    # Dynamically adjust the y-limits based on the random path
    max_price = np.max(random_path)
    min_price = np.min(random_path)
    buffer = (max_price - min_price) * 0.1  # Add some buffer space
    axs[0].set_ylim(min_price - buffer, max_price + buffer)

    axs[0].grid(True)
    axs[0].legend()

    # Plot average prices over all simulations on the right
    axs[1].plot(range(1, n_sim + 1), [np.mean(european_payoffs[:i]) for i in range(1, n_sim + 1)],
                label=r'\textbf{European Option Price}', color='blue')
    axs[1].plot(range(1, n_sim + 1), [np.mean(asian_payoffs[:i]) for i in range(1, n_sim + 1)],
                label=r'\textbf{Asian Option Price}', color='orange')

    axs[1].set_title(r'\textbf{Average Option Prices Over Simulations}', fontsize=16)
    axs[1].set_xlabel(r'\textbf{Simulation}', fontsize=14)
    axs[1].set_ylabel(r'\textbf{Price}', fontsize=14)
    axs[1].legend()
    axs[1].grid(True)

    # Save and display the final plot
    plt.tight_layout()
    final_plot_path = f"{save_dir}/random_simulation_side_by_side.png"
    plt.savefig(final_plot_path, dpi=300)
    plt.show()

    print(f"Final plot saved to {final_plot_path}")

    return avg_european, avg_asian


# Run simulation
np.random.seed(42)  # For reproducibility
european_price, asian_price = simulate_option_prices_random()

print(f"\nFinal Results:")
print(f"Average European Option Price: {european_price:.4f}")
print(f"Average Asian Option Price: {asian_price:.4f}")
print(f"Difference (European - Asian): {european_price - asian_price:.4f}")
