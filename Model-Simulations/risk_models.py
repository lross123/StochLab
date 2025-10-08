import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t

# --- functions from your snippet ---
def monte_carlo_var(returns, alpha=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - alpha) * len(sorted_returns))
    return -sorted_returns[index]

def monte_carlo_es(returns, alpha=0.95):
    sorted_returns = np.sort(returns)
    cutoff = int((1 - alpha) * len(sorted_returns))
    return -sorted_returns[:cutoff].mean()

def garch_simulation(alpha0, alpha1, beta1, T, r0=0.0, seed=None):
    rng = np.random.default_rng(seed)
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = alpha0 / (1 - alpha1 - beta1)
    for t in range(1, T):
        epsilon = rng.normal()
        sigma2[t] = alpha0 + alpha1 * (returns[t-1] ** 2) + beta1 * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * epsilon
    return returns, np.sqrt(sigma2)

def student_t_returns(df, size, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_t(df, size=size)


# ---------------- SIMULATION BLOCK ----------------
if __name__ == "__main__":
    np.random.seed(42)
    T = 1000

    # --- simulate GARCH(1,1) returns ---
    alpha0, alpha1, beta1 = 0.0001, 0.05, 0.9
    garch_returns, garch_sigma = garch_simulation(alpha0, alpha1, beta1, T)

    # --- generate Student-t returns (heavy-tailed) ---
    t_returns = student_t_returns(df=5, size=T)

    # --- combine (portfolio mix example) ---
    portfolio_returns = 0.6 * garch_returns + 0.4 * 0.02 * t_returns

    # --- compute VaR & ES ---
    alpha = 0.95
    VaR = monte_carlo_var(portfolio_returns, alpha)
    ES  = monte_carlo_es(portfolio_returns, alpha)

    print(f"95% Value-at-Risk: {VaR:.4f}")
    print(f"95% Expected Shortfall: {ES:.4f}")

    # --- 1️⃣ Return distribution histogram + VaR/ES lines ---
    plt.figure(figsize=(8,5))
    n, bins, _ = plt.hist(portfolio_returns, bins=50, density=True, alpha=0.6, label="Simulated returns")
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 300)
    plt.plot(x, norm.pdf(x, np.mean(portfolio_returns), np.std(portfolio_returns)),
             '--', label="Normal PDF fit")
    plt.axvline(-VaR, color="red", linestyle="--", label=f"VaR (95%) = {VaR:.3f}")
    plt.axvline(-ES,  color="purple", linestyle=":",  label=f"ES (95%) = {ES:.3f}")
    plt.title("Return Distribution with VaR and ES")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2️⃣ Cumulative returns ---
    cum_returns = np.cumsum(portfolio_returns)
    plt.figure(figsize=(8,4))
    plt.plot(cum_returns)
    plt.title("Cumulative Portfolio Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()

    # --- 3️⃣ Volatility clustering (from GARCH) ---
    plt.figure(figsize=(8,4))
    plt.plot(garch_sigma, label="Conditional σ_t")
    plt.title("GARCH(1,1) Volatility Clustering")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 4️⃣ QQ-plot vs Normal & t(5) ---
    import scipy.stats as stats
    plt.figure(figsize=(6,6))
    stats.probplot(portfolio_returns, dist=norm, plot=plt)
    plt.title("QQ-Plot vs Normal")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,6))
    stats.probplot(portfolio_returns, dist=student_t, sparams=(5,), plot=plt)
    plt.title("QQ-Plot vs Student-t(5)")
    plt.tight_layout()
    plt.show()
