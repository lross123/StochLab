import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# ----------------------------------------------------------------
# Brownian Motion and Geometric Brownian Motion classes (from your code)
# ----------------------------------------------------------------
from math import sqrt

class BrownianMotion:
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, N, x0=0.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        increments = rng.normal(0.0, sqrt(dt), N)
        dW = np.concatenate(([0], increments))
        W = np.cumsum(dW)
        x = x0 + self.mu * t + self.sigma * W
        return t, x

    def mean(self, t):
        return self.mu * t

    def variance(self, t):
        return (self.sigma ** 2) * t


class GeometricBrownianMotion:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, N, s0=1.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        increments = rng.normal(0.0, sqrt(dt), N)
        dW = np.concatenate(([0], increments))
        W = np.cumsum(dW)
        s = s0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        return t, s

    def mean(self, t, s0):
        return s0 * np.exp(self.mu * t)

    def variance(self, t, s0):
        return (s0**2) * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1)


# ----------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------
T = 1.0
N = 500
M = 5000     # Monte Carlo simulations
mu, sigma = 0.1, 0.3

bm = BrownianMotion(mu, sigma)
gbm = GeometricBrownianMotion(mu, sigma)

# ----------------------------------------------------------------
# 1️⃣ Simulate single sample paths
# ----------------------------------------------------------------
t, x_bm = bm.simulate(T, N, seed=42)
_, s_gbm = gbm.simulate(T, N, s0=1.0, seed=42)

plt.figure(figsize=(8,5))
plt.plot(t, x_bm, label="Brownian Motion (with drift)")
plt.plot(t, s_gbm, label="Geometric Brownian Motion")
plt.xlabel("Time")
plt.ylabel("Value / Price")
plt.title("Sample Paths of BM and GBM")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 2️⃣ Monte Carlo simulations for statistical properties
# ----------------------------------------------------------------
bm_final = np.zeros(M)
gbm_final = np.zeros(M)
rng = np.random.default_rng(123)

for i in range(M):
    _, x = bm.simulate(T, N, seed=rng.integers(1e6))
    _, s = gbm.simulate(T, N, s0=1.0, seed=rng.integers(1e6))
    bm_final[i] = x[-1]
    gbm_final[i] = s[-1]

# ----------------------------------------------------------------
# 3️⃣ Statistical metrics
# ----------------------------------------------------------------
bm_mean_emp, bm_var_emp = np.mean(bm_final), np.var(bm_final)
gbm_mean_emp, gbm_var_emp = np.mean(gbm_final), np.var(gbm_final)

bm_skew, bm_kurt = skew(bm_final), kurtosis(bm_final)
gbm_skew, gbm_kurt = skew(gbm_final), kurtosis(gbm_final)

bm_mean_theo = bm.mean(T)
bm_var_theo = bm.variance(T)

gbm_mean_theo = gbm.mean(T, 1.0)
gbm_var_theo = gbm.variance(T, 1.0)

print("---- Statistical Summary ----")
print(f"Brownian Motion:  mean_emp={bm_mean_emp:.4f}, var_emp={bm_var_emp:.4f}, "
      f"mean_theo={bm_mean_theo:.4f}, var_theo={bm_var_theo:.4f}")
print(f"                  skew={bm_skew:.4f}, kurtosis={bm_kurt:.4f}")
print(f"Geometric BM:     mean_emp={gbm_mean_emp:.4f}, var_emp={gbm_var_emp:.4f}, "
      f"mean_theo={gbm_mean_theo:.4f}, var_theo={gbm_var_theo:.4f}")
print(f"                  skew={gbm_skew:.4f}, kurtosis={gbm_kurt:.4f}")

# ----------------------------------------------------------------
# 4️⃣ Distribution plots
# ----------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(bm_final, bins=50, kde=True, label="Brownian Motion", color="skyblue", stat="density")
plt.axvline(bm_mean_theo, color="black", linestyle="--", label="Theoretical Mean")
plt.title("Distribution of BM Final Values X(T)")
plt.xlabel("X(T)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(gbm_final, bins=50, kde=True, label="Geometric Brownian Motion", color="orange", stat="density")
plt.axvline(gbm_mean_theo, color="black", linestyle="--", label="Theoretical Mean")
plt.title("Distribution of GBM Final Prices S(T)")
plt.xlabel("S(T)")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 5️⃣ Mean and variance convergence (theoretical vs empirical)
# ----------------------------------------------------------------
times = np.linspace(0, T, 20)
bm_emp_means, bm_emp_vars = [], []
gbm_emp_means, gbm_emp_vars = [], []

for ti in times:
    sample_means_bm, sample_means_gbm = [], []
    for _ in range(500):
        _, xb = bm.simulate(ti, N, seed=rng.integers(1e6))
        _, sg = gbm.simulate(ti, N, s0=1.0, seed=rng.integers(1e6))
        sample_means_bm.append(xb[-1])
        sample_means_gbm.append(sg[-1])
    bm_emp_means.append(np.mean(sample_means_bm))
    bm_emp_vars.append(np.var(sample_means_bm))
    gbm_emp_means.append(np.mean(sample_means_gbm))
    gbm_emp_vars.append(np.var(sample_means_gbm))

# Plot mean evolution
plt.figure(figsize=(8,4))
plt.plot(times, bm_emp_means, label="BM empirical mean")
plt.plot(times, [bm.mean(ti) for ti in times], "--", label="BM theoretical mean")
plt.plot(times, gbm_emp_means, label="GBM empirical mean")
plt.plot(times, [gbm.mean(ti,1.0) for ti in times], "--", label="GBM theoretical mean")
plt.title("Mean Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("E[X_t] or E[S_t]")
plt.legend()
plt.tight_layout()
plt.show()

# Plot variance evolution
plt.figure(figsize=(8,4))
plt.plot(times, bm_emp_vars, label="BM empirical var")
plt.plot(times, [bm.variance(ti) for ti in times], "--", label="BM theoretical var")
plt.plot(times, gbm_emp_vars, label="GBM empirical var")
plt.plot(times, [gbm.variance(ti,1.0) for ti in times], "--", label="GBM theoretical var")
plt.title("Variance Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Var[X_t] or Var[S_t]")
plt.legend()
plt.tight_layout()
plt.show()
