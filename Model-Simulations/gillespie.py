import numpy as np
import matplotlib.pyplot as plt


def gillespie(initial_state, propensity_func, update_func, t_max, seed=None):
    """
    Gillespie stochastic simulation algorithm (SSA).

    Args:
        initial_state: initial state vector
        propensity_func: function returning propensities (rates) given state
        update_func: function returning state change for a given reaction index
        t_max: simulation end time
        seed: random seed

    Returns:
        times: array of event times
        states: array of system states at each event
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    state = np.array(initial_state, dtype=float)
    times = [t]
    states = [state.copy()]

    while t < t_max:
        a = propensity_func(state)
        a0 = a.sum()
        if a0 <= 0:
            break
        r1 = rng.random()
        r2 = rng.random()
        tau = -np.log(r1) / a0
        t += tau
        # choose which reaction occurs
        cum_a = np.cumsum(a)
        reaction_index = np.searchsorted(cum_a, r2 * a0)
        state = state + update_func(reaction_index)
        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)


# --------------------------- TEST BLOCK ---------------------------

if __name__ == "__main__":
    """
    Example: Simple reversible reaction A ⇌ B
    A --(k1)--> B
    B --(k2)--> A
    """

    # ---- Parameters ----
    k1, k2 = 0.1, 0.05        # reaction rates
    A0, B0 = 100, 0           # initial molecule counts
    t_max = 100

    # ---- Define propensities and updates ----
    def propensity_func(state):
        A, B = state
        return np.array([k1 * A, k2 * B])  # a1 = k1*A, a2 = k2*B

    def update_func(reaction_index):
        # Reaction 0: A → B  (A -1, B +1)
        # Reaction 1: B → A  (A +1, B -1)
        updates = np.array([[-1, +1],
                            [+1, -1]])
        return updates[reaction_index]

    # ---- Run a single simulation ----
    times, states = gillespie([A0, B0], propensity_func, update_func, t_max, seed=42)
    A_traj, B_traj = states[:, 0], states[:, 1]

    print(f"Final state: A = {A_traj[-1]:.0f}, B = {B_traj[-1]:.0f}")
    print(f"Total number of events: {len(times)}")

    # ---- 1️⃣ Plot single realisation ----
    plt.figure(figsize=(7, 4))
    plt.step(times, A_traj, where="post", label="[A]")
    plt.step(times, B_traj, where="post", label="[B]")
    plt.xlabel("Time")
    plt.ylabel("Molecule count")
    plt.title("Gillespie SSA: A ⇌ B Reaction")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- 2️⃣ Multiple stochastic realisations ----
    n_runs = 20
    plt.figure(figsize=(7, 4))
    for i in range(n_runs):
        times_i, states_i = gillespie([A0, B0], propensity_func, update_func, t_max)
        plt.step(times_i, states_i[:, 0], where="post", alpha=0.4, lw=1)
    plt.xlabel("Time")
    plt.ylabel("Molecule count of A")
    plt.title("Stochastic Variability Across Realisations")
    plt.tight_layout()
    plt.show()

    # ---- 3️⃣ Histogram of final states ----
    final_As = []
    for i in range(500):
        _, states_i = gillespie([A0, B0], propensity_func, update_func, t_max)
        final_As.append(states_i[-1, 0])
    final_As = np.array(final_As)

    plt.figure(figsize=(6, 4))
    plt.hist(final_As, bins=20, color="cornflowerblue", edgecolor="black", density=True)
    plt.title("Distribution of [A] at Final Time")
    plt.xlabel("Final molecule count of A")
    plt.ylabel("Probability density")
    plt.tight_layout()
    plt.show()