import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def black_scholes_fd_cn(S_max, T, N_S, N_t, r, sigma, K,
                        option_type='call', theta=0.5):
    """
    Crank–Nicolson finite difference solver for European option pricing.

    Parameters:
      S_max      : maximum asset price on the grid
      T          : maturity
      N_S, N_t   : number of space/time grid points
      r, sigma   : risk–free rate and volatility
      K          : strike
      option_type: 'call' or 'put'
      theta      : scheme parameter (0.5 for Crank–Nicolson)

    Returns:
      S grid and full solution matrix V(t_j, S_i)
    """
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)
    V = np.zeros((N_t + 1, N_S + 1))

    # terminal payoff
    if option_type == 'call':
        V[-1] = np.maximum(S - K, 0)
    else:
        V[-1] = np.maximum(K - S, 0)

    # boundary conditions through time
    t_vec = np.linspace(0, T, N_t + 1)
    if option_type == 'call':
        V[:, 0] = 0
        V[:, -1] = S_max - K*np.exp(-r*(T - t_vec))
    else:
        V[:, 0]  = K*np.exp(-r*(T - t_vec))
        V[:, -1] = 0

    # precompute coefficients for interior nodes i=1..N_S-1
    i = np.arange(1, N_S)
    alpha = 0.5*dt*(sigma**2*i**2 - r*i)
    beta  = 1.0 + dt*(1 - theta)*(sigma**2*i**2 + r)
    gamma = 0.5*dt*(sigma**2*i**2 + r*i)

    # LHS (A) and RHS (B) diagonals
    A_lower = -theta*alpha[1:]     # length N_S-2
    A_diag  = 1.0 + theta*dt*(sigma**2*i**2 + r)
    A_upper = -theta*gamma[:-1]    # length N_S-2

    B_lower = (1 - theta)*alpha[1:]
    B_diag  = 1.0 - (1 - theta)*dt*(sigma**2*i**2 + r)
    B_upper = (1 - theta)*gamma[:-1]

    # tridiagonal solver
    def solve_tridiag(a, b, c, d):
        n = len(b)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        cp[0] = c[0]/b[0]
        dp[0] = d[0]/b[0]
        for k in range(1, n-1):
            denom = b[k] - a[k-1]*cp[k-1]
            cp[k] = c[k]/denom
            dp[k] = (d[k] - a[k-1]*dp[k-1])/denom
        dp[n-1] = (d[n-1] - a[n-2]*dp[n-2])/(b[n-1] - a[n-2]*cp[n-2])
        x = np.zeros(n)
        x[-1] = dp[-1]
        for k in range(n-2, -1, -1):
            x[k] = dp[k] - cp[k]*x[k+1]
        return x

    m = N_S - 1  # number of interior nodes
    for j in reversed(range(N_t)):
        U = V[j+1]                   # known solution at t_{j+1}
        rhs = np.zeros(m)
        # k = 1..m-2 (i = 2..N_S-2)
        if m > 2:
            rhs[1:-1] = (B_lower[:m-2]*U[1:m-1] +
                          B_diag[1:m-1]*U[2:m] +
                          B_upper[1:m-1]*U[3:m+1])
        # k = 0 (i = 1)
        rhs[0]  = B_diag[0]*U[1] + B_upper[0]*U[2]
        # k = m-1 (i = N_S-1)
        rhs[-1] = B_lower[-1]*U[N_S-2] + B_diag[-1]*U[N_S-1]
        # boundary contributions
        rhs[0]  += theta*alpha[0]*V[j,0]  + (1 - theta)*alpha[0]*V[j+1,0]
        rhs[-1] += theta*gamma[-1]*V[j,-1] + (1 - theta)*gamma[-1]*V[j+1,-1]
        # solve A * x = rhs
        V[j,1:-1] = solve_tridiag(A_lower, A_diag, A_upper, rhs)

    return S, V



# --------------------------- TEST / VISUALISATION BLOCK ---------------------------

if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    S_max = 300
    N_S, N_t = 200, 500
    option_type = "call"

    # Run solver
    S, V = black_scholes_fd_cn(S_max, T, N_S, N_t, r, sigma, K, option_type)
    V0 = V[0, :]

    # Analytical Black–Scholes price and Greeks
    def bs_analytical(S, K, T, r, sigma, opt='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return price, delta, gamma

    bs_price, bs_delta, bs_gamma = bs_analytical(S[1:], K, T, r, sigma, option_type)

    # Interpolate numerical price at S=K
    price_fd = np.interp(K, S, V0)
    price_analytical = bs_analytical(K, K, T, r, sigma, option_type)[0]
    abs_error = abs(price_fd - price_analytical)
    rmse = np.sqrt(np.mean((V0[1:] - bs_price)**2))

    print(f"Numerical price at S=K={K}: {price_fd:.4f}")
    print(f"Analytical price: {price_analytical:.4f}")
    print(f"Absolute error: {abs_error:.6f}")
    print(f"Grid RMSE vs analytical: {rmse:.6f}")

    # ---- 1️⃣ Option value vs S at t=0 ----
    plt.figure(figsize=(7, 4))
    plt.plot(S, V0, label="FD (CN)", color="royalblue")
    plt.plot(S[1:], bs_price, "--", label="Analytical", color="black")
    plt.axvline(K, color="gray", linestyle=":", label="Strike")
    plt.title("European Call Option Value at t=0")
    plt.xlabel("Asset Price S")
    plt.ylabel("Option Value V(S,0)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- 2️⃣ Greeks: Delta & Gamma ----
    dS = S[1] - S[0]
    delta_fd = np.gradient(V0, dS)
    gamma_fd = np.gradient(delta_fd, dS)

    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(S, delta_fd, label="Delta (FD)", color="teal")
    ax1.plot(S[1:], bs_delta, "--", color="black", label="Delta (Analytical)")
    ax1.set_xlabel("Asset Price S")
    ax1.set_ylabel("Delta")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(S, gamma_fd, label="Gamma (FD)", color="darkorange")
    ax2.plot(S[1:], bs_gamma, "--", color="gray", label="Gamma (Analytical)")
    ax2.set_ylabel("Gamma")
    ax2.legend(loc="upper right")
    plt.title("Delta and Gamma Comparison")
    plt.tight_layout()
    plt.show()

    # ---- 3️⃣ 3D price surface ----
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection="3d")
    t_vec = np.linspace(0, T, N_t + 1)
    S_mesh, T_mesh = np.meshgrid(S, t_vec)
    ax.plot_surface(S_mesh, T_mesh, V, cmap="viridis", linewidth=0)
    ax.set_title("Option Value Surface V(S,t)")
    ax.set_xlabel("Asset Price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Option Value")
    plt.tight_layout()
    plt.show()

    # ---- 4️⃣ Error heatmap ----
    analytical_full = np.zeros_like(V)
    for j, t in enumerate(np.linspace(0, T, N_t + 1)):
        analytical_full[j,1:] = bs_analytical(S[1:], K, T - t, r, sigma, option_type)[0]
    error = V - analytical_full
    plt.figure(figsize=(7,4))
    plt.imshow(np.abs(error), extent=[0, S_max, 0, T], origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(label="|Error|")
    plt.title("Absolute Error Heatmap (FD vs Analytical)")
    plt.xlabel("Asset Price S")
    plt.ylabel("Time t")
    plt.tight_layout()
    plt.show()