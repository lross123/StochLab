import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm, probplot

# ----------------------------------------------------------------
# Merton Jump-Diffusion process
# ----------------------------------------------------------------
from math import sqrt

class JumpDiffusion:
    def __init__(self, mu, sigma, lambda_, mu_j, sigma_j):
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def simulate(self, T, N, s0=1.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        s = np.zeros(N + 1)
        s[0] = s0

        for i in range(N):
            dw = rng.normal(0, sqrt(dt))
            diffusion = (self.mu - self.lambda_ * self.kappa) * s[i] * dt + self.sigma * s[i] * dw
            njumps = rng.poisson(self.lambda_ * dt)
            jump_factor = 1.0
            if njumps > 0:
                jump_sizes = np.exp(rng.normal(self.mu_j, self.sigma_j, size=njumps))
                jump_factor = np.prod(jump_sizes)
            s[i + 1] = s[i] + diffusion + s[i] * (jump_factor - 1)
        return t, s


# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------
mu = 0.10         # annual drift
sigma = 0.2       # diffusion volatility
lambda_ = 1.0     # average 1 jump per year
mu_j = -0.1       # average jump size (negative = downward)
sigma_j = 0.3     # jump size volatility
T = 2.0
N = 1000
M = 3000
s0 = 100

jd = JumpDiffusion(mu, sigma, lambda_, mu_j, sigma_j)

# ----------------------------------------------------------------
# 1️⃣ Simulate single path with jumps
# ----------------------------------------------------------------
t, s = jd.simulate(T, N, s0, seed=42)

plt.figure(figsize=(8,5))
plt.plot(t, s, label="Jump-Diffusion Path")
plt.title("Merton Jump-Diffusion Simulation (Single Path)")
plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# 2️⃣ Compare Jump-Diffusion vs GBM
# ----------------------------------------------------------------
def simulate_gbm(mu, sigma, T, N, s0=100, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = np.cumsum(rng.normal(0, sqrt(dt), N))
    S = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.concatenate(([0], W)))
    return t, S

t, s_gbm = simulate_gbm(mu, sigma, T, N, s0, seed=42)

plt.figure(figsize=(8,5))
plt.plot(t, s, label="Jump-Diffusion")
plt.plot(t, s_gbm, "--", label="Pure GBM (no jumps)")
plt.title("Jump-Diffusion vs Geometric Brownian Motion")
plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# 3️⃣ Monte Carlo Simulation for Statistical Analysis
# ----------------------------------------------------------------
rng = np.random.default_rng(123)
sT = np.zeros(M)
returns_all = []

for i in range(M):
    _, s_path = jd.simulate(T, N, s0, seed=rng.integers(1e6))
    sT[i] = s_path[-1]
    returns_all.extend(np.diff(np.log(s_path)))  # log returns

returns_all = np.array(returns_all)

# ----------------------------------------------------------------
# 4️⃣ Statistical metrics
# ----------------------------------------------------------------
mean_T = np.mean(sT)
var_T = np.var(sT)
skew_T = skew(sT)
kurt_T = kurtosis(sT)
mean_ret, std_ret = np.mean(returns_all), np.std(returns_all)
jump_freq = lambda_ * T  # expected number of jumps

print("---- Jump-Diffusion Statistical Summary ----")
print(f"Final price mean: {mean_T:.3f}, variance: {var_T:.3f}")
print(f"Return mean: {mean_ret:.5f}, std: {std_ret:.5f}")
print(f"Skewness: {skew_T:.3f}, Kurtosis: {kurt_T:.3f}")
print(f"Expected jumps per simulation: {jump_freq:.2f}")


# ----------------------------------------------------------------
# 5️⃣ Distribution of log returns
# ----------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(returns_all, bins=60, kde=True, color="royalblue", stat="density", label="Empirical")
x = np.linspace(returns_all.min(), returns_all.max(), 300)
plt.plot(x, norm.pdf(x, mean_ret, std_ret), 'k--', label="Normal Fit")
plt.title("Distribution of Log Returns (Jump-Diffusion)")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# QQ-plot of returns
plt.figure(figsize=(6,6))
probplot(returns_all, dist=norm, plot=plt)
plt.title("QQ-Plot of Log Returns vs Normal")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# 6️⃣ Volatility clustering visualization
# ----------------------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(returns_all[:1000])
plt.title("Volatility Clustering in Log Returns (First 1000 Steps)")
plt.xlabel("Time step")
plt.ylabel("Return")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# 7️⃣ Mean & variance convergence
# ----------------------------------------------------------------
times = np.linspace(0, T, 20)
means, vars_ = [], []

for ti in times:
    sims = []
    for _ in range(200):
        _, s_path = jd.simulate(ti, N, s0, seed=rng.integers(1e6))
        sims.append(s_path[-1])
    means.append(np.mean(sims))
    vars_.append(np.var(sims))

plt.figure(figsize=(8,4))
plt.plot(times, means, label="Empirical mean")
plt.title("Mean Evolution of Jump-Diffusion")
plt.xlabel("Time")
plt.ylabel("E[S_t]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(times, vars_, label="Empirical variance", color="darkorange")
plt.title("Variance Evolution of Jump-Diffusion")
plt.xlabel("Time")
plt.ylabel("Var[S_t]")
plt.legend()
plt.tight_layout()
plt.show()
