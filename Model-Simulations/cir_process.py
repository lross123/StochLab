import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm, probplot

# ---------------------------------------------------------------
# CIR process definition
# ---------------------------------------------------------------
from math import sqrt

class CIRProcess:
    def __init__(self, kappa: float, theta: float, sigma: float):
        self.kappa = kappa
        self.theta = theta
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
            sqrt_x = np.sqrt(max(x[i], 0.0))
            dw = rng.normal(0, sqrt(dt))
            x[i + 1] = x[i] + self.kappa * (self.theta - x[i]) * dt + self.sigma * sqrt_x * dw
            x[i + 1] = max(x[i + 1], 0.0)
        return t, x

    def stationary_mean(self):
        return self.theta

    def stationary_variance(self):
        return (self.sigma ** 2) * self.theta / (2 * self.kappa)

    def stationary_distribution(self):
        return self.stationary_mean(), self.stationary_variance()


# ---------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------
kappa = 1.5        # speed of mean reversion
theta = 0.04       # long-term mean (e.g., 4%)
sigma = 0.15       # volatility of the process
x0 = 0.02          # initial short rate
T = 5.0            # time horizon (years)
N = 1000           # steps
M = 5000           # number of Monte Carlo simulations

cir = CIRProcess(kappa, theta, sigma)

# ---------------------------------------------------------------
# 1️⃣ Simulate single path
# ---------------------------------------------------------------
t, x_path = cir.simulate(T, N, x0, seed=42)

plt.figure(figsize=(8,5))
plt.plot(t, x_path, label="CIR sample path")
plt.axhline(theta, color="k", linestyle="--", label="Long-run mean θ")
plt.title("CIR Process: Mean-Reverting Short Rate Path")
plt.xlabel("Time (years)")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 2️⃣ Monte Carlo simulation ensemble
# ---------------------------------------------------------------
rng = np.random.default_rng(123)
xT = np.zeros(M)
paths = []

for i in range(M):
    _, x = cir.simulate(T, N, x0, seed=rng.integers(1e6))
    xT[i] = x[-1]
    if i < 30:  # store first few paths for plotting
        paths.append(x)

# Plot multiple sample paths
plt.figure(figsize=(8,5))
for p in paths:
    plt.plot(t, p, color="gray", alpha=0.3)
plt.plot(t, np.mean(paths, axis=0), color="blue", lw=2, label="Mean of sample paths")
plt.axhline(theta, color="k", linestyle="--", label="Long-run mean θ")
plt.title("CIR Process: Multiple Sample Paths")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# 3️⃣ Empirical vs theoretical stationary stats
# ---------------------------------------------------------------
emp_mean = np.mean(xT)
emp_var = np.var(xT)
theo_mean, theo_var = cir.stationary_distribution()

skw = skew(xT)
krt = kurtosis(xT)
autocorr = np.corrcoef(x_path[:-1], x_path[1:])[0,1]

print("---- CIR Statistical Summary ----")
print(f"Theoretical mean: {theo_mean:.5f}, variance: {theo_var:.5f}")
print(f"Empirical mean:   {emp_mean:.5f}, variance: {emp_var:.5f}")
print(f"Skewness: {skw:.4f}, Kurtosis: {krt:.4f}, Lag-1 autocorrelation: {autocorr:.4f}")

# ---------------------------------------------------------------
# 4️⃣ Distribution of X(T)
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(xT, bins=50, kde=True, color="teal", stat="density", label="Empirical")
plt.axvline(theo_mean, color="black", linestyle="--", label="Theoretical Mean")
plt.title("Distribution of Terminal Values X(T)")
plt.xlabel("X(T)")
plt.legend()
plt.tight_layout()
plt.show()

# QQ-plot vs Normal
plt.figure(figsize=(6,6))
probplot(xT, dist=norm, plot=plt)
plt.title("QQ-Plot of X(T) vs Normal Distribution")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5️⃣ Mean & variance evolution across time
# ---------------------------------------------------------------
times = np.linspace(0, T, 20)
mean_emp, var_emp = [], []

for ti in times:
    sims = []
    for _ in range(500):
        _, x = cir.simulate(ti, N, x0, seed=rng.integers(1e6))
        sims.append(x[-1])
    mean_emp.append(np.mean(sims))
    var_emp.append(np.var(sims))

mean_theo = theta + (x0 - theta) * np.exp(-kappa * times)
var_theo = (sigma**2 / (2*kappa)) * (theta * (1 - np.exp(-kappa * times))**2 / kappa + np.exp(-kappa * times))

plt.figure(figsize=(8,4))
plt.plot(times, mean_emp, label="Empirical mean")
plt.plot(times, mean_theo, "--", label="Theoretical mean")
plt.axhline(theta, color="gray", linestyle=":", label="Long-run mean θ")
plt.title("Mean Reversion Over Time")
plt.xlabel("Time")
plt.ylabel("E[X_t]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(times, var_emp, label="Empirical variance")
plt.plot(times, var_theo, "--", label="Approx theoretical variance")
plt.title("Variance Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Var[X_t]")
plt.legend()
plt.tight_layout()
plt.show()
