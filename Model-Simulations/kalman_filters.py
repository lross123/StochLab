import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Core Filters (exactly as you wrote them)
# -------------------------------------------------------------------

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.x = np.asarray(x0)
        self.P = np.asarray(P0)

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        innovation = y - self.C @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        return self.x, self.P


class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, f, h, F_jac, H_jac, Q, R, x0, P0):
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.x = np.asarray(x0)
        self.P = np.asarray(P0)

    def predict(self, u):
        self.x = self.f(self.x, u)
        F = self.F_jac(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P

    def update(self, y):
        H = self.H_jac(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        innovation = y - self.h(self.x)
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        return self.x, self.P


class ParticleFilter:
    def __init__(self, n_particles, transition_func, likelihood_func, resample_thresh=0.5):
        self.n_particles = n_particles
        self.transition_func = transition_func
        self.likelihood_func = likelihood_func
        self.resample_thresh = resample_thresh
        self.particles = None
        self.weights = None

    def initialise(self, init_particles):
        self.particles = np.array(init_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self):
        self.particles = self.transition_func(self.particles)

    def update(self, observation):
        likelihoods = self.likelihood_func(observation, self.particles)
        self.weights *= likelihoods + 1e-300
        self.weights /= np.sum(self.weights)
        neff = 1.0 / np.sum(self.weights**2)
        if neff < self.resample_thresh * self.n_particles:
            self.resample()

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        idx = np.searchsorted(cumulative_sum, np.random.rand(self.n_particles))
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


# -------------------------------------------------------------------
# TEST / DEMONSTRATION BLOCK
# -------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    # -----------------------------------------------------------
    # 1️⃣ Linear Kalman Filter — Constant Velocity Tracking
    # -----------------------------------------------------------
    dt = 0.1
    A = np.array([[1, dt],
                  [0, 1]])
    B = np.zeros((2, 1))
    C = np.array([[1, 0]])   # measure position only
    Q = np.diag([1e-4, 1e-4])
    R = np.array([[0.05]])
    x0 = np.array([0, 1])
    P0 = np.eye(2) * 0.1

    kf = KalmanFilter(A, B, C, Q, R, x0, P0)

    # Simulate ground truth
    N = 100
    true_x = np.zeros((N, 2))
    true_x[0] = x0
    for t in range(1, N):
        true_x[t] = A @ true_x[t-1]
    measurements = true_x[:, 0] + np.random.normal(0, np.sqrt(R[0, 0]), N)

    # Run Kalman Filter
    estimates = []
    for y in measurements:
        kf.predict(u=np.array([0]))
        x_est, _ = kf.update(np.array([y]))
        estimates.append(x_est)
    estimates = np.array(estimates)

    # ---- Plot results ----
    plt.figure(figsize=(7, 4))
    plt.plot(true_x[:, 0], label="True Position", color="black")
    plt.plot(measurements, ".", alpha=0.4, label="Measurements", color="grey")
    plt.plot(estimates[:, 0], label="KF Estimate", color="teal")
    plt.fill_between(range(N),
                     estimates[:, 0] - 2 * np.sqrt(kf.P[0, 0]),
                     estimates[:, 0] + 2 * np.sqrt(kf.P[0, 0]),
                     color="teal", alpha=0.2, label="±2σ bound")
    plt.title("Kalman Filter — Position Tracking")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # 2️⃣ Extended Kalman Filter — Nonlinear Sinusoidal Tracking
    # -----------------------------------------------------------
    f = lambda x, u: np.array([x[0] + 0.05 * np.sin(x[0])])   # nonlinear state
    h = lambda x: np.array([np.sin(x[0])])                     # nonlinear measurement
    F_jac = lambda x, u: np.array([[1 + 0.05 * np.cos(x[0])]])
    H_jac = lambda x: np.array([[np.cos(x[0])]])
    Q = np.array([[1e-3]])
    R = np.array([[0.05]])
    x0 = np.array([0.1])
    P0 = np.array([[0.1]])

    ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R, x0, P0)

    # True dynamics + noisy measurements
    N = 80
    x_true = np.zeros(N)
    y_obs = np.zeros(N)
    x_true[0] = 0.1
    for t in range(1, N):
        x_true[t] = f(np.array([x_true[t-1]]), np.array([0])) + np.random.normal(0, np.sqrt(Q[0, 0]))
        y_obs[t] = h(np.array([x_true[t]])) + np.random.normal(0, np.sqrt(R[0, 0]))

    estimates = np.zeros(N)
    for t in range(N):
        ekf.predict(np.array([0]))
        x_est, _ = ekf.update(np.array([y_obs[t]]))
        estimates[t] = x_est[0]

    # ---- Plot results ----
    plt.figure(figsize=(7, 4))
    plt.plot(x_true, label="True state", color="black")
    plt.plot(y_obs, ".", label="Observations", alpha=0.5)
    plt.plot(estimates, label="EKF estimate", color="darkorange")
    plt.title("Extended Kalman Filter — Nonlinear Tracking")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # 3️⃣ Particle Filter — Nonlinear Example
    # -----------------------------------------------------------
    n_particles = 500
    transition_func = lambda x: x + 0.05 * np.sin(x) + np.random.normal(0, 0.05, size=x.shape)
    likelihood_func = lambda y, x: np.exp(-0.5 * ((y - np.sin(x.flatten())) ** 2) / 0.05**2)
    pf = ParticleFilter(n_particles, transition_func, likelihood_func)
    pf.initialise(np.random.normal(0, 1, size=n_particles))

    estimates_pf = []
    for t in range(N):
        pf.predict()
        pf.update(y_obs[t])
        estimates_pf.append(pf.estimate())

    estimates_pf = np.array(estimates_pf).flatten()

    # ---- Plot results ----
    plt.figure(figsize=(7, 4))
    plt.plot(x_true, label="True state", color="black")
    plt.plot(y_obs, ".", label="Observations", alpha=0.5)
    plt.plot(estimates_pf, label="Particle Filter estimate", color="mediumseagreen")
    plt.title("Particle Filter — Nonlinear State Estimation")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Optional: visualise particles at final time ----
    plt.figure(figsize=(6, 4))
    plt.hist(pf.particles, bins=30, color="lightgreen", edgecolor="black")
    plt.title("Particle Distribution at Final Time")
    plt.xlabel("State value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()