import numpy as np
import matplotlib.pyplot as plt


class MertonJumpDiffusion:
    """
    Merton Jump Diffusion process.
    Combines Brownian diffusion with random jumps governed by a Poisson process.
    """
    def __init__(self, mu, sigma, lambda_, mu_j, sigma_j):
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1

    def simulate(self, T, N, s0=1.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        s = np.zeros(N + 1)
        s[0] = s0
        for i in range(N):
            dw = rng.normal(0, np.sqrt(dt))
            # diffusion part
            diffusion = (self.mu - self.lambda_ * self.kappa) * s[i] * dt + self.sigma * s[i] * dw
            # jump part
            njumps = rng.poisson(self.lambda_ * dt)
            jump_factor = 1.0
            if njumps > 0:
                jump_sizes = np.exp(rng.normal(self.mu_j, self.sigma_j, size=njumps))
                jump_factor = np.prod(jump_sizes)
            s[i + 1] = s[i] + diffusion + s[i] * (jump_factor - 1)
        return t, s


class VarianceGamma:
    """
    Variance Gamma process using difference of two gamma processes.
    """
    def __init__(self, sigma, nu, theta):
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

    def simulate(self, T, N, s0=1.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        s = np.zeros(N + 1)
        s[0] = s0
        for i in range(N):
            # time change via gamma increment
            G = rng.gamma(dt / self.nu, self.nu)
            dX = self.theta * G + self.sigma * np.sqrt(G) * rng.normal()
            s[i + 1] = s[i] * np.exp(dX)
        return t, s


class NormalInverseGaussian:
    """
    Normal Inverse Gaussian (NIG) process simulation based on subordination.
    """
    def __init__(self, alpha, beta, delta, mu):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu

    def simulate(self, T, N, s0=1.0, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0, T, N + 1)
        s = np.zeros(N + 1)
        s[0] = s0
        for i in range(N):
            # Inverse Gaussian subordinator parameters
            nu = self.delta * dt
            gamma = np.sqrt(self.alpha**2 - self.beta**2)
            mu_ig = nu / gamma
            # Michael–Schucany–Haas method
            w = rng.normal()
            x = mu_ig + (mu_ig**2 * w**2) / (2) - (mu_ig / 2) * np.sqrt(4 * mu_ig * w**2 + (mu_ig * w**2)**2)
            u = rng.random()
            tau = x if u <= mu_ig / (mu_ig + x) else mu_ig**2 / x
            # increment
            dX = self.mu * dt + self.beta * tau + np.sqrt(tau) * rng.normal()
            s[i + 1] = s[i] * np.exp(dX)
        return t, s


# --------------------------- TEST BLOCK ---------------------------

if __name__ == "__main__":
    # Simulation parameters
    T, N, s0 = 1.0, 500, 100.0

    # 1️⃣ Merton Jump Diffusion
    mjd = MertonJumpDiffusion(mu=0.1, sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.3)
    t1, s1 = mjd.simulate(T, N, s0=s0)
    plt.plot(t1, s1, label="Merton Jump Diffusion")

    # 2️⃣ Variance Gamma
    vg = VarianceGamma(sigma=0.2, nu=0.2, theta=0.1)
    t2, s2 = vg.simulate(T, N, s0=s0)
    plt.plot(t2, s2, label="Variance Gamma")

    # 3️⃣ Normal Inverse Gaussian
    nig = NormalInverseGaussian(alpha=3, beta=-1, delta=0.1, mu=0.05)
    t3, s3 = nig.simulate(T, N, s0=s0)
    plt.plot(t3, s3, label="Normal Inverse Gaussian")

    # 📈 Visualise
    plt.title("Sample Paths of Lévy-Type Processes")
    plt.xlabel("Time")
    plt.ylabel("Asset Price")
    plt.legend()
    plt.tight_layout()
    plt.show()