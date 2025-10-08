import numpy as np
import numpy.linalg as la

class MarkovChain:
    """
    Discrete-time Markov chain defined by a transition matrix.
    """
    def __init__(self, transition_matrix: np.ndarray):
        self.P = np.asarray(transition_matrix)
        self.n = self.P.shape[0]
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError('Rows of transition matrix must sum to 1')

    def stationary_distribution(self):
        """
        Compute the stationary distribution by solving pi P = pi.
        """
        eigvals, eigvecs = la.eig(self.P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()
        return pi

    def hitting_time_distribution(self, start_state, target_state, n_sim=10000):
        """
        Estimate distribution of hitting times from start_state to target_state using simulation.
        """
        hits = []
        rng = np.random.default_rng()
        for _ in range(n_sim):
            state = start_state
            t = 0
            while state != target_state and t < 1000:
                state = rng.choice(self.n, p=self.P[state])
                t += 1
            hits.append(t)
        return np.array(hits)

    def expected_return(self, rewards, start_state, n_steps):
        """
        Compute expected cumulative rewards over n_steps starting from start_state.
        """
        rewards = np.asarray(rewards)
        # probability distribution after k steps: start distribution times P^k
        expected_rewards = 0
        dist = np.zeros(self.n)
        dist[start_state] = 1
        for k in range(n_steps):
            expected_rewards += dist @ rewards
            dist = dist @ self.P
        return expected_rewards
    
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Example transition matrix (2-state Markov chain)
    P = np.array([
        [0.9, 0.1],
        [0.5, 0.5]
    ])

    mc = MarkovChain(P)

    # ---- 1️⃣ Stationary distribution ----
    pi = mc.stationary_distribution()
    print("Stationary distribution:", pi)

    # Bar plot for stationary probabilities
    plt.figure(figsize=(5, 4))
    plt.bar(range(mc.n), pi, color='steelblue', edgecolor='black')
    plt.title("Stationary Distribution")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.xticks(range(mc.n))
    plt.tight_layout()
    plt.show()


    # ---- 2️⃣ Probability evolution over time ----
    n_steps = 20
    dist = np.zeros((n_steps + 1, mc.n))
    dist[0] = [1, 0]  # start in state 0
    for t in range(n_steps):
        dist[t + 1] = dist[t] @ P

    plt.figure(figsize=(6, 4))
    plt.plot(range(n_steps + 1), dist[:, 0], label="P(state=0)")
    plt.plot(range(n_steps + 1), dist[:, 1], label="P(state=1)")
    plt.axhline(pi[0], color='C0', ls='--', lw=1)
    plt.axhline(pi[1], color='C1', ls='--', lw=1)
    plt.title("Convergence of State Probabilities")
    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ---- 3️⃣ Hitting time distribution ----
    hits = mc.hitting_time_distribution(0, 1)
    print("\nMean hitting time from 0 → 1:", hits.mean())

    plt.figure(figsize=(6, 4))
    plt.hist(hits, bins=50, color='lightcoral', edgecolor='black', density=True)
    plt.title("Distribution of Hitting Times (0 → 1)")
    plt.xlabel("Hitting time (steps)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


    # ---- 4️⃣ Expected return trajectory ----
    rewards = np.array([1, 2])
    expected_cum_reward = [mc.expected_return(rewards, 0, k) for k in range(1, 21)]

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 21), expected_cum_reward, marker='o', color='darkgreen')
    plt.title("Expected Cumulative Reward vs Steps")
    plt.xlabel("Number of steps")
    plt.ylabel("Expected cumulative reward")
    plt.tight_layout()
    plt.show()