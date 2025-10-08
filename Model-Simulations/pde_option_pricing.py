import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def black_scholes_fd(S_max, T, N_S, N_t, r, sigma, K, option_type='call'):
    """
    Finite difference solver for the Black–Scholes PDE (explicit scheme).
    """
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)
    V = np.zeros((N_t + 1, N_S + 1))

    # terminal payoff
    if option_type == 'call':
        V[-1, :] = np.maximum(S - K, 0)
    elif option_type == 'put':
        V[-1, :] = np.maximum(K - S, 0)
    else:
        raise ValueError("Unknown option type")

    # boundary conditions
    if option_type == 'call':
        V[:, 0] = 0
        V[:, -1] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N_t + 1)))
    else:
        V[:, 0] = K * np.exp(-r * (T - np.linspace(0, T, N_t + 1)))
        V[:, -1] = 0

    # explicit finite difference scheme
    for j in reversed(range(N_t)):
        for i in range(1, N_S):
            a = 0.5 * dt * (sigma ** 2 * i ** 2 - r * i)
            b = 1 - dt * (sigma ** 2 * i ** 2 + r)
            c = 0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
            V[j, i] = a * V[j + 1, i - 1] + b * V[j + 1, i] + c * V[j + 1, i + 1]

    return S, V


# --------------------------- TEST / DEMONSTRATION BLOCK ---------------------------

if __name__ == "__main__":
    # Parameters
    S_max = 200
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    N_S = 200
    N_t = 1000
    option_type = 'call'

    # Run finite difference solver
    S_grid, V_grid = black_scholes_fd(S_max, T, N_S, N_t, r, sigma, K, option_type)
    V0 = V_grid[0, :]   # option values at t=0

    # Analytical Black–Scholes price
    def black_scholes_analytical(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    analytical = black_scholes_analytical(S_grid[1:], K, T, r, sigma, option_type)
    fd_price_at_K = np.interp(K, S_grid, V0)
    analytical_at_K = black_scholes_analytical(K, K, T, r, sigma, option_type)

    print(f"Finite Difference price at S=K={K}: {fd_price_at_K:.4f}")
    print(f"Analytical Black–Scholes price:     {analytical_at_K:.4f}")
    print(f"Absolute error:                     {abs(fd_price_at_K - analytical_at_K):.6f}")

    # ---- 1️⃣ Plot option value vs S at t=0 ----
    plt.figure(figsize=(7, 4))
    plt.plot(S_grid, V0, label="Finite Difference", color="royalblue")
    plt.plot(S_grid[1:], analytical, "--", label="Analytical", color="black")
    plt.title(f"European {option_type.capitalize()} Option Price at t=0")
    plt.xlabel("Asset Price S")
    plt.ylabel("Option Value V(S, 0)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- 2️⃣ 3D surface of V(S, t) ----
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    t_vals = np.linspace(0, T, N_t + 1)
    S_mesh, T_mesh = np.meshgrid(S_grid, t_vals)
    surf = ax.plot_surface(S_mesh, T_mesh, V_grid, cmap="viridis", linewidth=0)
    ax.set_xlabel("Asset Price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Option Value V")
    ax.set_title("Finite Difference Solution Surface")
    plt.tight_layout()
    plt.show()

    # ---- 3️⃣ Heatmap of option values ----
    plt.figure(figsize=(7, 4))
    plt.imshow(V_grid, extent=[0, S_max, 0, T], origin="lower", aspect="auto", cmap="plasma")
    plt.colorbar(label="Option Value")
    plt.title("Option Value Heatmap (V(S, t))")
    plt.xlabel("Asset Price S")
    plt.ylabel("Time t")
    plt.tight_layout()
    plt.show()

    # ---- 4️⃣ Convergence test (optional) ----
    N_values = [50, 100, 200, 400]
    errors = []
    for N in N_values:
        Sg, Vg = black_scholes_fd(S_max, T, N, N_t, r, sigma, K, option_type)
        price_fd = np.interp(K, Sg, Vg[0, :])
        errors.append(abs(price_fd - analytical_at_K))
    plt.figure(figsize=(6, 4))
    plt.plot(N_values, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Convergence of Finite Difference Price at S=K")
    plt.xlabel("Number of Space Grid Points (log)")
    plt.ylabel("Absolute Error (log)")
    plt.tight_layout()
    plt.show()