import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Euler–Maruyama, Milstein, Heun schemes
# -----------------------------
def euler_maruyama(mu_func, sigma_func, x0, T, N, seed=None):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0
    rng = np.random.default_rng(seed)
    for i in range(N):
        ti, xi = t[i], x[i]
        dw = rng.normal(0, np.sqrt(dt))
        x[i + 1] = xi + mu_func(xi, ti) * dt + sigma_func(xi, ti) * dw
    return t, x


def milstein(mu_func, sigma_func, dsigma_dx, x0, T, N, seed=None):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0
    rng = np.random.default_rng(seed)
    for i in range(N):
        ti, xi = t[i], x[i]
        dw = rng.normal(0, np.sqrt(dt))
        x[i + 1] = xi + mu_func(xi, ti) * dt + sigma_func(xi, ti) * dw \
                   + 0.5 * sigma_func(xi, ti) * dsigma_dx(xi, ti) * (dw**2 - dt)
    return t, x


def heun(mu_func, sigma_func, x0, T, N, seed=None):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0
    rng = np.random.default_rng(seed)
    for i in range(N):
        ti, xi = t[i], x[i]
        dw = rng.normal(0, np.sqrt(dt))
        # predictor step
        x_pred = xi + mu_func(xi, ti) * dt + sigma_func(xi, ti) * dw
        # corrector step
        mu_avg = 0.5 * (mu_func(xi, ti) + mu_func(x_pred, ti + dt))
        x[i + 1] = xi + mu_avg * dt + sigma_func(xi, ti) * dw
    return t, x


# -----------------------------
# 2️⃣ Test case: Geometric Brownian Motion
# dX = μ X dt + σ X dW
# Analytical solution: X_t = X_0 exp((μ - 0.5σ²)t + σ W_t)
# -----------------------------
mu = 0.1
sigma = 0.3
x0 = 1.0
T = 1.0
N = 500

mu_func = lambda x, t: mu * x
sigma_func = lambda x, t: sigma * x
dsigma_dx = lambda x, t: sigma

# -----------------------------
# 3️⃣ Simulate single sample paths
# -----------------------------
t, x_euler = euler_maruyama(mu_func, sigma_func, x0, T, N, seed=42)
_, x_milstein = milstein(mu_func, sigma_func, dsigma_dx, x0, T, N, seed=42)
_, x_heun = heun(mu_func, sigma_func, x0, T, N, seed=42)

# Analytical mean path for comparison
x_true_mean = x0 * np.exp(mu * t)

# Plot sample paths
plt.figure(figsize=(8, 5))
plt.plot(t, x_euler, label="Euler–Maruyama", alpha=0.8)
plt.plot(t, x_milstein, label="Milstein", alpha=0.8)
plt.plot(t, x_heun, label="Heun", alpha=0.8)
plt.plot(t, x_true_mean, "k--", label="Analytical Mean E[X_t]")
plt.title("SDE Simulation Comparison (GBM)")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 4️⃣ Monte Carlo: distribution at final time
# -----------------------------
M = 5000  # number of Monte Carlo paths
xT_euler = np.zeros(M)
xT_milstein = np.zeros(M)
xT_heun = np.zeros(M)
rng = np.random.default_rng(123)

for m in range(M):
    _, x1 = euler_maruyama(mu_func, sigma_func, x0, T, N, seed=rng.integers(1e6))
    _, x2 = milstein(mu_func, sigma_func, dsigma_dx, x0, T, N, seed=rng.integers(1e6))
    _, x3 = heun(mu_func, sigma_func, x0, T, N, seed=rng.integers(1e6))
    xT_euler[m], xT_milstein[m], xT_heun[m] = x1[-1], x2[-1], x3[-1]

# Analytical mean and variance
true_mean = x0 * np.exp(mu * T)
true_var = (x0**2) * np.exp(2*mu*T) * (np.exp(sigma**2*T) - 1)

# Plot histograms
plt.figure(figsize=(8,5))
plt.hist(xT_euler, bins=40, alpha=0.5, label="Euler–Maruyama")
plt.hist(xT_milstein, bins=40, alpha=0.5, label="Milstein")
plt.hist(xT_heun, bins=40, alpha=0.5, label="Heun")
plt.axvline(true_mean, color="k", linestyle="--", label="Analytical Mean")
plt.title("Distribution of X(T) Across 5000 Simulations")
plt.xlabel("X(T)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 5️⃣ Error metrics: mean & variance convergence
# -----------------------------
means = [np.mean(xT_euler), np.mean(xT_milstein), np.mean(xT_heun)]
vars_  = [np.var(xT_euler), np.var(xT_milstein), np.var(xT_heun)]

methods = ["Euler", "Milstein", "Heun"]
plt.figure(figsize=(6,4))
plt.bar(methods, means, label="Simulated Mean", alpha=0.6)
plt.axhline(true_mean, color="k", linestyle="--", label="Analytical Mean")
plt.title("Mean Convergence")
plt.ylabel("E[X_T]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(methods, vars_, label="Simulated Variance", alpha=0.6, color="orange")
plt.axhline(true_var, color="k", linestyle="--", label="Analytical Var")
plt.title("Variance Convergence")
plt.ylabel("Var[X_T]")
plt.legend()
plt.tight_layout()
plt.show()

# Print metrics
print(f"Analytical E[X_T] = {true_mean:.4f}, Var[X_T] = {true_var:.4f}")
for m, muhat, varhat in zip(methods, means, vars_):
    print(f"{m:<10}:  mean = {muhat:.4f},  var = {varhat:.4f}")
