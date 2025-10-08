import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class MeanReversionStrategy:
    """
    Simple pairs trading strategy assuming the spread follows an Ornstein–Uhlenbeck process.
    """
    def __init__(self, prices1, prices2):
        self.prices1 = np.array(prices1)
        self.prices2 = np.array(prices2)

    def estimate_parameters(self):
        log_spread = np.log(self.prices1) - np.log(self.prices2)
        dt = 1.0
        y = log_spread[1:]
        x = log_spread[:-1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        theta = -np.log(slope) / dt
        mu = intercept / (1 - slope)
        residuals = y - (slope * x + intercept)
        sigma = np.std(residuals) * np.sqrt(2 * theta / (1 - slope ** 2))
        return theta, mu, sigma

    def backtest(self, entry_z=1.0, exit_z=0.0):
        log_spread = np.log(self.prices1) - np.log(self.prices2)
        mean = np.mean(log_spread)
        std = np.std(log_spread)
        zscores = (log_spread - mean) / std
        positions = np.zeros(len(zscores))
        for i in range(1, len(zscores)):
            if positions[i-1] == 0:
                if zscores[i] > entry_z:
                    positions[i] = -1
                elif zscores[i] < -entry_z:
                    positions[i] = 1
            else:
                if abs(zscores[i]) < exit_z:
                    positions[i] = 0
                else:
                    positions[i] = positions[i-1]
        returns1 = np.diff(self.prices1) / self.prices1[:-1]
        returns2 = np.diff(self.prices2) / self.prices2[:-1]
        pnl = positions[:-1] * (returns1 - returns2)
        cumulative = np.cumsum(pnl)
        return positions, pnl, cumulative, zscores


# --------------------------- TEST / DEMO BLOCK ---------------------------

if __name__ == "__main__":
    np.random.seed(0)

    # ---- 1️⃣ Simulate synthetic mean-reverting spread ----
    T, N = 1.0, 500
    dt = T / N
    theta_true, mu_true, sigma_true = 1.5, 0.0, 0.1
    spread = np.zeros(N)
    for t in range(1, N):
        spread[t] = spread[t-1] + theta_true*(mu_true - spread[t-1])*dt + sigma_true*np.sqrt(dt)*np.random.randn()

    # Create synthetic prices
    base = 100 * np.exp(0.0002*np.arange(N))
    prices1 = base * np.exp(spread/2)
    prices2 = base * np.exp(-spread/2)

    # ---- 2️⃣ Estimate OU parameters ----
    strat = MeanReversionStrategy(prices1, prices2)
    theta_est, mu_est, sigma_est = strat.estimate_parameters()

    print("Estimated OU parameters:")
    print(f"  θ (mean reversion rate): {theta_est:.3f}")
    print(f"  μ (long-term mean):       {mu_est:.3f}")
    print(f"  σ (volatility):           {sigma_est:.3f}")
    print(f"True values: θ={theta_true}, μ={mu_true}, σ={sigma_true}")

    # ---- 3️⃣ Backtest strategy ----
    positions, pnl, cumulative, zscores = strat.backtest(entry_z=1.0, exit_z=0.2)

    # ---- 4️⃣ Plot the spread and z-score ----
    log_spread = np.log(prices1) - np.log(prices2)
    mean = np.mean(log_spread)
    std = np.std(log_spread)

    plt.figure(figsize=(7, 4))
    plt.plot(zscores, label="Z-score of spread", color="steelblue")
    plt.axhline(1.0, color="red", linestyle="--", label="Entry +1σ")
    plt.axhline(-1.0, color="red", linestyle="--")
    plt.axhline(0.0, color="black", linestyle=":")
    plt.title("Spread Z-score and Trading Thresholds")
    plt.xlabel("Time step")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- 5️⃣ Plot trading positions ----
    plt.figure(figsize=(7, 3))
    plt.plot(positions, color="darkorange")
    plt.title("Trading Positions (1 = Long, -1 = Short)")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.show()

    # ---- 6️⃣ Plot cumulative P&L ----
    plt.figure(figsize=(7, 4))
    plt.plot(cumulative, color="green")
    plt.title("Cumulative P&L (Mean Reversion Strategy)")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative return")
    plt.tight_layout()
    plt.show()

    # ---- 7️⃣ Show OU spread simulation ----
    plt.figure(figsize=(7, 4))
    plt.plot(spread, label="Simulated OU Spread", color="teal")
    plt.axhline(mu_true, color="black", linestyle="--", label="Mean")
    plt.title("Ornstein–Uhlenbeck Process (True Spread)")
    plt.xlabel("Time step")
    plt.ylabel("Spread value")
    plt.legend()
    plt.tight_layout()
    plt.show()