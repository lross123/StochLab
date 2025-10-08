import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis, probplot

# ---------------------------------------------------------------
# Ornstein–Uhlenbeck Process Definition
# ---------------------------------------------------------------
class OrnsteinUhlenbeck:
    def __init__(self, theta: float, mu: float, sigma: float):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, N: int, x0: float = 0.0, seed: int | None = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        dt = T / N
        t = np.linspace(0, T, N + 1)
        x = np.zeros(N + 1)
        x[0] = x0
        for i in range(N):
            dw = rng.normal(0, np.sqrt(dt))
            x[i + 1] = x[i] + self.theta * (self.mu - x[i]) * dt + self.sigma * dw
        return t, x

    def stationary_mean(self):
        return self.mu

    def stationary_variance(self):
        return (self.sigma ** 2) / (2 * self.theta)

    def stationary_distribution(self):
        return self.stationary_mean(), self.stationary_variance()


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
theta = 2.0     # speed of mean reversion
mu = 0.5        # long-run mean
sigma = 0.3     # volatility
x0 = 0.0        # initial value
T = 5.0         # time horizon
N = 1000        # number of steps
M = 3000        # number of simulations

ou = OrnsteinUhlenbeck(theta, mu, sigma)

# ---------------------------------------------------------------
# 1️⃣ Single path simulation
# ---------------------------------------------------------------
t, x_path = ou.simulate(T, N, x0, seed=42)

plt.figure(figsize=(8,5))
plt.plot(t, x_path, label="OU Path")
plt.axhline(mu, color="black", linestyle="--", label="Long-run mean μ")
plt.title("Ornstein–Uhlenbeck Process: Single Sample Path")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 2️⃣ Multiple paths for visualisation
# ---------------------------------------------------------------
rng = np.random.default_rng(123)
paths = []
for i in range(30):
    _, x = ou.simulate(T, N, x0, seed=rng.integers(1e6))
    paths.append(x)

plt.figure(figsize=(8,5))
for p in paths:
    plt.plot(t, p, color="gray", alpha=0.3)
plt.plot(t, np.mean(paths, axis=0), color="teal", lw=2, label="Empirical Mean Path")
plt.axhline(mu, color="k", linestyle="--", label="Long-run Mean μ")
plt.title("Multiple Ornstein–Uhlenbeck Sample Paths")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 3️⃣ Monte Carlo terminal distribution
# ---------------------------------------------------------------
xT = np.zeros(M)
for i in range(M):
    _, x = ou.simulate(T, N, x0, seed=rng.integers(1e6))
    xT[i] = x[-1]

emp_mean = np.mean(xT)
emp_var = np.var(xT)
skw = skew(xT)
krt = kurtosis(xT)
autocorr = np.corrcoef(x_path[:-1], x_path[1:])[0,1]

theo_mean, theo_var = ou.stationary_distribution()

print("---- OU Statistical Summary ----")
print(f"Theoretical mean: {theo_mean:.4f}, variance: {theo_var:.5f}")
print(f"Empirical mean:   {emp_mean:.4f}, variance: {emp_var:.5f}")
print(f"Skewness: {skw:.4f}, Kurtosis: {krt:.4f}, Lag-1 autocorrelation: {autocorr:.4f}")


# ---------------------------------------------------------------
# 4️⃣ Distribution & QQ-plot
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(xT, bins=50, kde=True, stat="density", color="slateblue", label="Empirical X(T)")
plt.axvline(theo_mean, color="black", linestyle="--", label="Theoretical Mean μ")
plt.title("Distribution of Terminal Values X(T)")
plt.xlabel("X(T)")
plt.legend()
plt.tight_layout()
plt.show()

# QQ-plot vs Normal
plt.figure(figsize=(6,6))
probplot(xT, dist=norm, plot=plt)
plt.title("QQ-Plot of X(T) vs Normal")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 5️⃣ Mean & variance convergence over time
# ---------------------------------------------------------------
times = np.linspace(0, T, 20)
means, vars_ = [], []

for ti in times:
    sims = []
    for _ in range(500):
        _, x = ou.simulate(ti, N, x0, seed=rng.integers(1e6))
        sims.append(x[-1])
    means.append(np.mean(sims))
    vars_.append(np.var(sims))

mean_theo = mu + (x0 - mu) * np.exp(-theta * times)
var_theo = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * times))

plt.figure(figsize=(8,4))
plt.plot(times, means, label="Empirical Mean")
plt.plot(times, mean_theo, "--", label="Theoretical Mean")
plt.axhline(mu, color="gray", linestyle=":", label="Long-run mean μ")
plt.title("Mean Reversion Over Time")
plt.xlabel("Time")
plt.ylabel("E[X_t]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(times, vars_, label="Empirical Variance", color="darkorange")
plt.plot(times, var_theo, "--", color="black", label="Theoretical Variance")
plt.title("Variance Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Var[X_t]")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 6️⃣ Autocorrelation decay visualization
# ---------------------------------------------------------------
max_lag = 50
acf = [np.corrcoef(x_path[:-lag], x_path[lag:])[0,1] for lag in range(1, max_lag)]
plt.figure(figsize=(7,4))
plt.stem(range(1, max_lag), acf, basefmt=" ")
plt.title("Autocorrelation Function of OU Process")
plt.xlabel("Lag")
plt.ylabel("ρ(lag)")
plt.tight_layout()
plt.show()
