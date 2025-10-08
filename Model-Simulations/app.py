"""
Flask application providing interactive dashboards for various stochastic models.

This web server exposes a simple HTML dashboard and a REST API.  Users can adjust
model parameters on the front‑end and fetch new simulation results via the API.
Plotly.js is used client‑side to render the resulting charts.  The back‑end
computes data by delegating to functions implemented in the provided Python
modules (e.g. SDE solvers, option pricing, Kalman filters, etc.).

To run the application locally, install Flask and run this script.  Then
navigate to http://localhost:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np

from sde_solvers import euler_maruyama, milstein, heun
from monte_carlo_engine import OptionPricingEngine
from kalman_filters import KalmanFilter
from gillespie import gillespie as gillespie_func
from markov_chain import MarkovChain
from mean_reversion_strategy import MeanReversionStrategy
from pde_option_pricing import black_scholes_fd
from black_scholes import black_scholes_fd_cn
from risk_models import monte_carlo_var, monte_carlo_es, garch_simulation, student_t_returns
from jump_diffusion_models import MertonJumpDiffusion, VarianceGamma, NormalInverseGaussian

# -----------------------------------------------------------------------------
# Minimal stochastic process classes for the interactive dashboard
#
# Several of the provided example modules (e.g. brownian_motion.py,
# ornstein_uhlenbeck.py, cir_process.py) include heavy plotting dependencies
# such as seaborn.  Those libraries are not guaranteed to be installed in the
# runtime environment, so importing directly from those files may fail.  To
# ensure the API can still simulate sample paths, we re‑implement the core
# stochastic process classes here with no plotting dependencies.

from math import sqrt


class BrownianMotion:
    """Simple Brownian motion with optional drift and diffusion parameters."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, N: int, x0: float = 0.0, seed: int | None = None):
        import numpy as np
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        # simulate Brownian increments
        increments = rng.normal(0.0, sqrt(dt), N)
        dW = np.concatenate(([0.0], increments))
        W = np.cumsum(dW)
        x = x0 + self.mu * t + self.sigma * W
        return t, x


class GeometricBrownianMotion:
    """Geometric Brownian motion for modelling asset prices."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, N: int, s0: float = 1.0, seed: int | None = None):
        import numpy as np
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        increments = rng.normal(0.0, sqrt(dt), N)
        dW = np.concatenate(([0.0], increments))
        W = np.cumsum(dW)
        s = s0 * np.exp((self.mu - 0.5 * self.sigma ** 2) * t + self.sigma * W)
        return t, s


class OrnsteinUhlenbeck:
    """Ornstein–Uhlenbeck mean‑reverting process."""

    def __init__(self, theta: float, mu: float, sigma: float) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, N: int, x0: float = 0.0, seed: int | None = None):
        import numpy as np
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        x = np.zeros(N + 1)
        x[0] = x0
        for i in range(N):
            dw = rng.normal(0.0, sqrt(dt))
            x[i + 1] = x[i] + self.theta * (self.mu - x[i]) * dt + self.sigma * dw
        return t, x


class CIRProcess:
    """Cox–Ingersoll–Ross process for non‑negative mean‑reverting dynamics."""

    def __init__(self, kappa: float, theta: float, sigma: float) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate(self, T: float, N: int, x0: float = 0.0, seed: int | None = None):
        import numpy as np
        rng = np.random.default_rng(seed)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        x = np.zeros(N + 1)
        x[0] = x0
        for i in range(N):
            sqrt_x = sqrt(max(x[i], 0.0))
            dw = rng.normal(0.0, sqrt(dt))
            x[i + 1] = x[i] + self.kappa * (self.theta - x[i]) * dt + self.sigma * sqrt_x * dw
            # enforce non‑negativity
            x[i + 1] = max(x[i + 1], 0.0)
        return t, x


app = Flask(__name__, template_folder='templates')


@app.route('/')
def index() -> str:
    """Render the main dashboard page."""
    return render_template('index.html')


@app.route('/api/sde')
def api_sde():
    """Simulate a simple one‑dimensional SDE using various numerical schemes.

    Query parameters:
        method  – integration method ('euler', 'milstein', or 'heun')
        mu      – drift coefficient (float)
        sigma   – diffusion coefficient (float)
        x0      – initial value (float)
        T       – terminal time (float)
        N       – number of timesteps (int)

    Returns:
        JSON with arrays of time points and simulated values.
    """
    method = request.args.get('method', 'euler').lower()
    mu = float(request.args.get('mu', 0.1))
    sigma = float(request.args.get('sigma', 0.3))
    x0 = float(request.args.get('x0', 1.0))
    T = float(request.args.get('T', 1.0))
    N = max(1, int(request.args.get('N', 500)))
    # define drift and diffusion functions for geometric Brownian motion
    mu_func = lambda x, t: mu * x
    sigma_func = lambda x, t: sigma * x
    if method == 'euler':
        t_array, x_array = euler_maruyama(mu_func, sigma_func, x0, T, N)
    elif method == 'milstein':
        dsigma_dx = lambda x, t: sigma  # constant derivative for GBM
        t_array, x_array = milstein(mu_func, sigma_func, dsigma_dx, x0, T, N)
    elif method == 'heun':
        t_array, x_array = heun(mu_func, sigma_func, x0, T, N)
    else:
        return jsonify({'error': 'Unknown SDE method'}), 400
    return jsonify({'t': t_array.tolist(), 'x': x_array.tolist()})


@app.route('/api/mc_option')
def api_mc_option():
    """Price a European call option via Monte Carlo and return payoff distribution.

    Query parameters:
        mu       – drift of the underlying GBM (float)
        sigma    – volatility of the underlying GBM (float)
        r        – risk‑free rate (float)
        s0       – initial asset price (float)
        K        – strike price (float)
        T        – maturity (float)
        N_paths  – number of Monte Carlo sample paths (int)
        N_steps  – time steps per path (int)

    Returns:
        JSON with the discounted payoff distribution and point estimates.
    """
    mu = float(request.args.get('mu', 0.05))
    sigma = float(request.args.get('sigma', 0.2))
    r = float(request.args.get('r', 0.05))
    s0 = float(request.args.get('s0', 100.0))
    K = float(request.args.get('K', 100.0))
    T = float(request.args.get('T', 1.0))
    N_paths = max(1, int(request.args.get('N_paths', 1000)))
    N_steps = max(1, int(request.args.get('N_steps', 252)))

    # define a simple GBM process compatible with OptionPricingEngine
    class GBMProcess:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
        def simulate(self, T, N_steps, s0=s0, seed=None):
            rng = np.random.default_rng(seed)
            dt = T / N_steps
            t = np.linspace(0, T, N_steps + 1)
            s = np.zeros(N_steps + 1)
            s[0] = s0
            for i in range(N_steps):
                dw = rng.normal(0, np.sqrt(dt))
                s[i + 1] = s[i] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * dw)
            return t, s

    process = GBMProcess(mu, sigma)
    engine = OptionPricingEngine(process)
    payoff_func = lambda s: max(s - K, 0)
    mean, stderr, discounted, _ = engine.price_european_option(
        payoff_func, T, N_paths, N_steps, r, s0
    )
    bs_price = engine.black_scholes_price(s0, K, T, r, sigma, option_type='call')
    return jsonify({
        'discounted_payoffs': discounted.tolist(),
        'mean': float(mean),
        'stderr': float(stderr),
        'bs_price': float(bs_price)
    })


@app.route('/api/kalman')
def api_kalman():
    """Simulate a simple constant‑velocity tracking scenario and apply a Kalman filter.

    Query parameters:
        dt    – time step size (float)
        q     – process noise variance (float)
        r     – measurement noise variance (float)
        N     – number of time steps (int)

    Returns:
        JSON with arrays of true positions, noisy measurements, Kalman estimates and variance.
    """
    dt = float(request.args.get('dt', 0.1))
    q_var = float(request.args.get('q', 1e-4))
    r_var = float(request.args.get('r', 0.05))
    N = max(1, int(request.args.get('N', 100)))

    # state space: [position, velocity]
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [0]])
    C = np.array([[1, 0]])
    Q = np.diag([q_var, q_var])
    R = np.array([[r_var]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.1

    kf = KalmanFilter(A, B, C, Q, R, x0, P0)

    # generate ground truth and noisy measurements
    true_x = np.zeros((N, 2))
    true_x[0] = x0
    for i in range(1, N):
        true_x[i] = A @ true_x[i - 1]
    rng = np.random.default_rng()
    measurements = true_x[:, 0] + rng.normal(0, np.sqrt(r_var), size=N)

    estimates = np.zeros(N)
    variances = np.zeros(N)
    for i in range(N):
        kf.predict(u=np.array([0]))
        x_est, P_est = kf.update(np.array([measurements[i]]))
        estimates[i] = x_est[0]
        variances[i] = P_est[0, 0]

    return jsonify({
        't': list(range(N)),
        'true': true_x[:, 0].tolist(),
        'measurements': measurements.tolist(),
        'estimates': estimates.tolist(),
        'variance': variances.tolist()
    })


@app.route('/api/gillespie')
def api_gillespie():
    """Run the Gillespie stochastic simulation algorithm for a simple reversible reaction.

    Query parameters:
        k1      – forward reaction rate (float)
        k2      – backward reaction rate (float)
        A0      – initial count of species A (int)
        B0      – initial count of species B (int)
        t_max   – final simulation time (float)
        runs    – number of simulations for final distribution (int, optional)

    Returns:
        JSON with time points and species counts for a single realisation.  If runs > 1,
        also returns the distribution of final A counts across multiple realisations.
    """
    k1 = float(request.args.get('k1', 0.1))
    k2 = float(request.args.get('k2', 0.05))
    A0 = float(request.args.get('A0', 100.0))
    B0 = float(request.args.get('B0', 0.0))
    t_max = float(request.args.get('t_max', 100.0))
    runs = int(request.args.get('runs', 1))

    # define propensities and updates for a reversible reaction A ⇌ B
    def propensity(state):
        A, B = state
        return np.array([k1 * A, k2 * B])
    def update(idx):
        return np.array([[-1, 1], [1, -1]])[idx]

    times, states = gillespie_func([A0, B0], propensity, update, t_max)
    result = {
        'times': times.tolist(),
        'A': states[:, 0].tolist(),
        'B': states[:, 1].tolist()
    }
    if runs > 1:
        final_As = []
        for _ in range(runs):
            _, states_i = gillespie_func([A0, B0], propensity, update, t_max)
            final_As.append(states_i[-1, 0])
        result['final_As'] = final_As
    return jsonify(result)


@app.route('/api/markov')
def api_markov():
    """Analyse a two‑state Markov chain given its transition matrix.

    Query parameters:
        P00, P01, P10, P11 – transition probabilities for a 2×2 matrix
        n_steps            – number of steps to simulate state probabilities

    Returns:
        JSON with the stationary distribution, the evolution of state probabilities
        over time (starting from state 0), and a sample of hitting times from state 0 to 1.
    """
    try:
        P00 = float(request.args.get('P00', 0.9))
        P01 = float(request.args.get('P01', 0.1))
        P10 = float(request.args.get('P10', 0.5))
        P11 = float(request.args.get('P11', 0.5))
    except ValueError:
        return jsonify({'error': 'Transition probabilities must be numeric'}), 400
    n_steps = max(1, int(request.args.get('n_steps', 20)))
    P = np.array([[P00, P01], [P10, P11]])
    try:
        mc = MarkovChain(P)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    pi = mc.stationary_distribution()
    # simulate distribution evolution starting from state 0
    dist = np.array([1.0, 0.0])
    dist_over_time = [dist.tolist()]
    for _ in range(n_steps):
        dist = dist @ P
        dist_over_time.append(dist.tolist())
    hits = mc.hitting_time_distribution(0, 1, n_sim=1000)
    return jsonify({
        'stationary': pi.tolist(),
        'dist_time': dist_over_time,
        'hits': hits.tolist()
    })


@app.route('/api/meanreversion')
def api_meanreversion():
    """Simulate a synthetic mean‑reverting spread and backtest a pairs trading strategy.

    Query parameters:
        theta    – mean reversion rate (float)
        mu       – long‑run mean (float)
        sigma    – volatility of the spread (float)
        T        – time horizon (float)
        N        – number of time steps (int)
        entry_z  – z‑score threshold to enter trades (float)
        exit_z   – z‑score threshold to exit trades (float)

    Returns:
        JSON containing the z‑scores of the spread, trading positions, instantaneous
        P&L, cumulative P&L, and estimated OU parameters (θ̂, μ̂, σ̂).
    """
    theta = float(request.args.get('theta', 1.5))
    mu = float(request.args.get('mu', 0.0))
    sigma = float(request.args.get('sigma', 0.1))
    T = float(request.args.get('T', 1.0))
    N = max(2, int(request.args.get('N', 500)))
    entry_z = float(request.args.get('entry_z', 1.0))
    exit_z = float(request.args.get('exit_z', 0.0))
    dt = T / N
    rng = np.random.default_rng()
    spread = np.zeros(N)
    for i in range(1, N):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    base = 100 * np.exp(0.0002 * np.arange(N))
    prices1 = base * np.exp(spread / 2)
    prices2 = base * np.exp(-spread / 2)
    strat = MeanReversionStrategy(prices1, prices2)
    theta_est, mu_est, sigma_est = strat.estimate_parameters()
    positions, pnl, cumulative, zscores = strat.backtest(entry_z=entry_z, exit_z=exit_z)
    return jsonify({
        'z_scores': zscores.tolist(),
        'positions': positions.tolist(),
        'pnl': pnl.tolist(),
        'cum_pnl': cumulative.tolist(),
        'theta_est': float(theta_est),
        'mu_est': float(mu_est),
        'sigma_est': float(sigma_est)
    })


@app.route('/api/jumpdiffusion')
def api_jumpdiffusion():
    """Simulate a jump or Lévy process path.

    Query parameters:
        model    – which model to use ('merton', 'vg', 'nig')
        T        – time horizon (float)
        N        – number of time steps (int)
        s0       – initial price (float)
        Additional parameters depend on the chosen model:
            merton: mu, sigma, lambda, mu_j, sigma_j
            vg    : sigma, nu, theta
            nig   : alpha, beta, delta, mu

    Returns:
        JSON with arrays of times and simulated asset prices.
    """
    model = request.args.get('model', 'merton').lower()
    T = float(request.args.get('T', 1.0))
    N = max(1, int(request.args.get('N', 500)))
    s0 = float(request.args.get('s0', 100.0))
    if model == 'merton':
        mu = float(request.args.get('mu', 0.1))
        sigma = float(request.args.get('sigma', 0.2))
        lambda_ = float(request.args.get('lambda', 0.5))
        mu_j = float(request.args.get('mu_j', -0.1))
        sigma_j = float(request.args.get('sigma_j', 0.3))
        proc = MertonJumpDiffusion(mu, sigma, lambda_, mu_j, sigma_j)
    elif model == 'vg':
        sigma = float(request.args.get('sigma', 0.2))
        nu = float(request.args.get('nu', 0.2))
        theta = float(request.args.get('theta', 0.1))
        proc = VarianceGamma(sigma, nu, theta)
    elif model == 'nig':
        alpha = float(request.args.get('alpha', 3.0))
        beta = float(request.args.get('beta', -1.0))
        delta = float(request.args.get('delta', 0.1))
        mu = float(request.args.get('mu', 0.05))
        proc = NormalInverseGaussian(alpha, beta, delta, mu)
    else:
        return jsonify({'error': 'Unknown jump diffusion model'}), 400
    t, s = proc.simulate(T, N, s0=s0)
    return jsonify({'t': t.tolist(), 's': s.tolist()})


@app.route('/api/pde')
def api_pde():
    """Solve the Black–Scholes PDE via an explicit finite difference scheme.

    Query parameters:
        S_max      – maximum asset price (float)
        T          – maturity (float)
        N_S, N_t   – number of space and time grid points (int)
        r, sigma   – risk‑free rate and volatility (float)
        K          – strike price (float)
        option_type– 'call' or 'put'

    Returns:
        JSON with the asset price grid and option values at t=0.
    """
    S_max = float(request.args.get('S_max', 200.0))
    T_val = float(request.args.get('T', 1.0))
    N_S = max(1, int(request.args.get('N_S', 200)))
    N_t = max(1, int(request.args.get('N_t', 200)))
    r = float(request.args.get('r', 0.05))
    sigma = float(request.args.get('sigma', 0.2))
    K = float(request.args.get('K', 100.0))
    option_type = request.args.get('option_type', 'call')
    try:
        S, V = black_scholes_fd(S_max, T_val, N_S, N_t, r, sigma, K, option_type)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    V0 = V[0, :]
    return jsonify({'S': S.tolist(), 'V0': V0.tolist()})


@app.route('/api/black_scholes')
def api_black_scholes():
    """Solve the Black–Scholes PDE using the Crank–Nicolson scheme implemented in black_scholes.py.

    Query parameters are analogous to /api/pde.  See the docstring of
    black_scholes.black_scholes_fd_cn for details.
    """
    S_max = float(request.args.get('S_max', 300.0))
    T_val = float(request.args.get('T', 1.0))
    N_S = max(1, int(request.args.get('N_S', 200)))
    N_t = max(1, int(request.args.get('N_t', 500)))
    r = float(request.args.get('r', 0.05))
    sigma = float(request.args.get('sigma', 0.2))
    K = float(request.args.get('K', 100.0))
    option_type = request.args.get('option_type', 'call')
    try:
        S, V = black_scholes_fd_cn(S_max, T_val, N_S, N_t, r, sigma, K, option_type)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    V0 = V[0, :]
    return jsonify({'S': S.tolist(), 'V0': V0.tolist()})


@app.route('/api/risk')
def api_risk():
    """Compute portfolio return distribution and risk measures using a mixture of GARCH and Student‑t returns.

    Query parameters:
        alpha    – confidence level for VaR/ES (float)
        T        – number of return observations (int)
        alpha0   – GARCH(1,1) alpha0 parameter (float)
        alpha1   – GARCH alpha1 parameter (float)
        beta1    – GARCH beta1 parameter (float)
        df       – degrees of freedom for Student‑t returns (int)

    Returns:
        JSON containing the simulated returns, conditional volatility sequence,
        and Value‑at‑Risk and Expected Shortfall estimates.
    """
    alpha = float(request.args.get('alpha', 0.95))
    T_val = max(1, int(request.args.get('T', 1000)))
    alpha0 = float(request.args.get('alpha0', 0.0001))
    alpha1 = float(request.args.get('alpha1', 0.05))
    beta1 = float(request.args.get('beta1', 0.9))
    df = max(1, int(request.args.get('df', 5)))
    g_returns, g_sigma = garch_simulation(alpha0, alpha1, beta1, T_val)
    t_returns = student_t_returns(df, T_val)
    # portfolio returns: weighted sum of GARCH and Student‑t components
    port_returns = 0.6 * g_returns + 0.4 * 0.02 * t_returns
    VaR = monte_carlo_var(port_returns, alpha)
    ES = monte_carlo_es(port_returns, alpha)
    return jsonify({
        'returns': port_returns.tolist(),
        'volatility': g_sigma.tolist(),
        'VaR': float(VaR),
        'ES': float(ES)
    })


# -----------------------------------------------------------------------------
# Additional API endpoints for basic stochastic processes
#
# These endpoints provide sample path simulation for Brownian motion,
# geometric Brownian motion, Ornstein–Uhlenbeck and CIR processes.  They
# complement the existing SDE, mean reversion and jump diffusion routes.


@app.route('/api/brownian')
def api_brownian() -> 'flask.Response':
    """Simulate a Brownian or Geometric Brownian motion path.

    Query parameters:
        type   – 'bm' for Brownian motion or 'gbm' for geometric Brownian (str)
        mu     – drift coefficient (float)
        sigma  – diffusion volatility (float)
        x0     – initial value for Brownian motion (float)
        s0     – initial price for GBM (float)
        T      – time horizon (float)
        N      – number of time steps (int)

    Returns:
        JSON with arrays of time points and simulated values.
    """
    model_type = request.args.get('type', 'bm').lower()
    mu = float(request.args.get('mu', 0.0))
    sigma = float(request.args.get('sigma', 1.0))
    x0 = float(request.args.get('x0', 0.0))
    s0 = float(request.args.get('s0', 1.0))
    T = float(request.args.get('T', 1.0))
    N = max(1, int(request.args.get('N', 500)))
    if model_type == 'bm':
        proc = BrownianMotion(mu, sigma)
        t, x = proc.simulate(T, N, x0=x0)
        return jsonify({'t': t.tolist(), 'x': x.tolist()})
    elif model_type == 'gbm':
        proc = GeometricBrownianMotion(mu, sigma)
        t, s = proc.simulate(T, N, s0=s0)
        return jsonify({'t': t.tolist(), 's': s.tolist()})
    else:
        return jsonify({'error': 'Unknown Brownian model type'}), 400


@app.route('/api/ou')
def api_ou() -> 'flask.Response':
    """Simulate an Ornstein–Uhlenbeck process path.

    Query parameters:
        theta – speed of mean reversion (float)
        mu    – long‑run mean (float)
        sigma – volatility parameter (float)
        x0    – initial value (float)
        T     – time horizon (float)
        N     – number of steps (int)

    Returns:
        JSON with arrays of time points and simulated values.
    """
    theta = float(request.args.get('theta', 1.0))
    mu = float(request.args.get('mu', 0.0))
    sigma = float(request.args.get('sigma', 0.3))
    x0 = float(request.args.get('x0', 0.0))
    T = float(request.args.get('T', 1.0))
    N = max(1, int(request.args.get('N', 500)))
    proc = OrnsteinUhlenbeck(theta, mu, sigma)
    t, x = proc.simulate(T, N, x0=x0)
    return jsonify({'t': t.tolist(), 'x': x.tolist()})


@app.route('/api/cir')
def api_cir() -> 'flask.Response':
    """Simulate a Cox–Ingersoll–Ross process path.

    Query parameters:
        kappa – speed of mean reversion (float)
        theta – long‑term mean (float)
        sigma – volatility parameter (float)
        x0    – initial value (float)
        T     – time horizon (float)
        N     – number of steps (int)

    Returns:
        JSON with arrays of time points and simulated values.
    """
    kappa = float(request.args.get('kappa', 1.0))
    theta = float(request.args.get('theta', 0.04))
    sigma = float(request.args.get('sigma', 0.1))
    x0 = float(request.args.get('x0', 0.0))
    T = float(request.args.get('T', 1.0))
    N = max(1, int(request.args.get('N', 500)))
    proc = CIRProcess(kappa, theta, sigma)
    t, x = proc.simulate(T, N, x0=x0)
    return jsonify({'t': t.tolist(), 'x': x.tolist()})


if __name__ == '__main__':
    # Only run the web server if this script is executed directly.
    app.run(host='0.0.0.0', port=5001, debug=False)
    
    
    
    
    
    
    