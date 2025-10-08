import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class OptionPricingEngine:
    def __init__(self, process):
        self.process = process

    def price_european_option(self, payoff_func, T, N_paths, N_steps, r, s0):
        payoffs = np.zeros(N_paths)
        for i in range(N_paths):
            t, s = self.process.simulate(T, N_steps, s0=s0)
            payoffs[i] = payoff_func(s[-1])
        discounted = np.exp(-r * T) * payoffs
        mean = discounted.mean()
        stderr = discounted.std(ddof=1) / np.sqrt(N_paths)
        return mean, stderr, discounted, payoffs

    def price_european_option_antithetic(self, payoff_func, T, N_paths, N_steps, r, s0):
        half_paths = N_paths // 2
        payoffs = np.zeros(2 * half_paths)
        for i in range(half_paths):
            t, s = self.process.simulate(T, N_steps, s0=s0)
            t2, s_ant = self.process.simulate(T, N_steps, s0=s0, seed=i)
            payoffs[i] = payoff_func(s[-1])
            payoffs[half_paths + i] = payoff_func(s_ant[-1])
        discounted = np.exp(-r * T) * payoffs
        mean = discounted.mean()
        stderr = discounted.std(ddof=1) / np.sqrt(N_paths)
        return mean, stderr, discounted

    @staticmethod
    def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
        """Analytical Black–Scholes benchmark."""
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "call":
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# --------------------------- TEST BLOCK ---------------------------

if __name__ == "__main__":
    # --- 1️⃣ Define process ---
    class GBMProcess:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma

        def simulate(self, T, N_steps, s0=100, seed=None):
            np.random.seed(seed)
            dt = T / N_steps
            t = np.linspace(0, T, N_steps + 1)
            W = np.random.normal(0, np.sqrt(dt), size=N_steps)
            S = np.zeros(N_steps + 1)
            S[0] = s0
            for i in range(1, N_steps + 1):
                S[i] = S[i - 1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * W[i - 1]
                )
            return t, S

    # --- 2️⃣ Parameters ---
    mu, sigma, S0, K, T, r = 0.05, 0.2, 100, 100, 1, 0.05
    N_paths, N_steps = 10000, 252

    # --- 3️⃣ Setup pricing engine ---
    process = GBMProcess(mu, sigma)
    engine = OptionPricingEngine(process)
    payoff_func = lambda s: max(s - K, 0)

    # --- 4️⃣ Monte Carlo estimation ---
    price, stderr, discounted, payoffs = engine.price_european_option(
        payoff_func, T, N_paths, N_steps, r, S0
    )

    bs_price = engine.black_scholes_price(S0, K, T, r, sigma, option_type="call")

    print(f"Monte Carlo European Call Option Price: {price:.4f} ± {1.96*stderr:.4f}")
    print(f"Analytical Black–Scholes Price:         {bs_price:.4f}")
    print(f"Absolute error:                         {abs(price - bs_price):.4f}")

    # --- 5️⃣ High-level visualisations ---

    # (a) Sample asset price paths
    plt.figure(figsize=(6, 4))
    for i in range(10):
        t, s = process.simulate(T, N_steps, s0=S0)
        plt.plot(t, s, lw=1)
    plt.title("Sample GBM Asset Price Paths")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.tight_layout()
    plt.show()

    # (b) Distribution of discounted payoffs
    plt.figure(figsize=(6, 4))
    plt.hist(discounted, bins=50, color="lightblue", edgecolor="black", density=True)
    plt.axvline(price, color="red", linestyle="--", label="Mean payoff")
    plt.title("Distribution of Discounted Payoffs")
    plt.xlabel("Discounted payoff")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (c) Monte Carlo convergence
    running_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(running_means, color="teal", label="Monte Carlo estimate")
    plt.axhline(bs_price, color="black", linestyle="--", label="Black–Scholes benchmark")
    plt.title("Monte Carlo Convergence to Black–Scholes Price")
    plt.xlabel("Number of simulated paths")
    plt.ylabel("Option price estimate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (d) Cumulative error plot
    abs_error = np.abs(running_means - bs_price)
    plt.figure(figsize=(6, 4))
    plt.plot(abs_error, color="darkorange")
    plt.yscale("log")
    plt.title("Convergence Error (log scale)")
    plt.xlabel("Number of simulated paths")
    plt.ylabel("|MC − BS|")
    plt.tight_layout()
    plt.show()