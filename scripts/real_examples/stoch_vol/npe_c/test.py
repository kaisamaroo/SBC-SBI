import numpy as np
import torch
from torch.distributions import constraints
import scipy.stats
from math import sqrt
from sbi.utils import process_prior
from sbi.utils.user_input_checks import process_simulator
from sbi.inference import NPE_C, simulate_for_sbi
import pickle, yaml, os, time, argparse
from pathlib import Path


def static_params_prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
                            rho_lower, rho_upper):
    sigma2 = scipy.stats.invgamma.rvs(a=sigma2_alpha, scale=sigma2_beta)
    beta2  = scipy.stats.invgamma.rvs(a=beta2_alpha,  scale=beta2_beta)
    rho    = scipy.stats.uniform.rvs(loc=rho_lower, scale=rho_upper - rho_lower)
    return sigma2, beta2, rho


def latent_trajectory_prior_rv(sigma2, rho, T, initial_distribution_variance=0.01):
    zs = []
    for k in range(T + 1):
        if k == 0:
            z = scipy.stats.norm.rvs(loc=0, scale=sqrt(initial_distribution_variance))
        else:
            z = scipy.stats.norm.rvs(loc=rho * zs[k - 1], scale=sqrt(sigma2))
        zs.append(z)
    return np.array(zs)


def prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
             rho_lower, rho_upper, T, initial_distribution_variance=0.01):
    sigma2, beta2, rho = static_params_prior_rv(
        sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper)
    z = latent_trajectory_prior_rv(sigma2, rho, T, initial_distribution_variance)
    return np.concatenate([np.array([sigma2, beta2, rho]), z])  # (T+4,)


# ─────────────────────────────────────────────────────────────
# PRIOR CLASS  — plain Python class, NOT a Distribution subclass
#
# theta = [sigma2, beta2, rho, z_0, z_1,..., z_T]  shape: (T+4,)
#
#   sigma2 ~ InvGamma(sigma2_alpha, sigma2_beta)       > 0
#   beta2  ~ InvGamma(beta2_alpha,  beta2_beta)        > 0
#   rho    ~ Uniform(rho_lower, rho_upper)             in (-1, 1) typically
#   z_0    ~ Normal(0, sqrt(initial_distribution_variance))
#   z_k    ~ Normal(rho * z_{k-1}, sqrt(sigma2))       k = 1..T
#
# process_prior() will wrap this into a sbi-compatible
# PyTorch Distribution using the lower/upper bounds we pass.
# ─────────────────────────────────────────────────────────────

class SVMPrior:
    """
    Plain-class hierarchical prior for the stochastic volatility model.
    Implements.sample() and.log_prob() — the two methods process_prior
    requires. Returns numpy arrays from.sample() so that process_prior
    sets prior_returns_numpy=True and process_simulator handles the
    numpy<->tensor casting automatically.
    """

    def __init__(self, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
                 rho_lower, rho_upper, T, initial_distribution_variance=0.01):
        self.sigma2_alpha = sigma2_alpha
        self.sigma2_beta  = sigma2_beta
        self.beta2_alpha  = beta2_alpha
        self.beta2_beta   = beta2_beta
        self.rho_lower    = rho_lower
        self.rho_upper    = rho_upper
        self.T            = T
        self.initial_distribution_variance = initial_distribution_variance

    # ----------------------------------------------------------
    def sample(self, sample_shape=torch.Size([])) -> np.ndarray:
        """
        Returns numpy array of shape (*sample_shape, T+4).
        Returning numpy is intentional — process_prior detects this
        and sets prior_returns_numpy=True, which tells process_simulator
        to cast theta to numpy before calling the simulator.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        elif isinstance(sample_shape, torch.Size):
            sample_shape = tuple(sample_shape)

        N = int(np.prod(sample_shape)) if len(sample_shape) > 0 else 1

        theta = np.stack([
            prior_rv(
                self.sigma2_alpha, self.sigma2_beta,
                self.beta2_alpha,  self.beta2_beta,
                self.rho_lower,    self.rho_upper,
                self.T,
                initial_distribution_variance=self.initial_distribution_variance,
            )
            for _ in range(N)
        ])  # (N, T+4)

        if len(sample_shape) == 0:
            return theta.squeeze(0)           # (T+4,)
        return theta.reshape(*sample_shape, self.T + 4)

    # ----------------------------------------------------------
    def log_prob(self, theta) -> torch.Tensor:
        """
        log p(theta) = log p(sigma2)
                     + log p(beta2)
                     + log p(rho)
                     + log p(z_0)
                     + sum_k log p(z_k | z_{k-1}, sigma2, rho)

        Args:
            theta: Tensor or ndarray of shape (N, T+4) or (T+4,)

        Returns:
            log_prob: Tensor of shape (N,)
        """
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        theta = np.atleast_2d(theta)  # (N, T+4)

        if theta.shape[-1] != self.T + 4:
            return torch.full((theta.shape[0],), float('-inf'), dtype=torch.float32)

        sigma2 = theta[:, 0]   # (N,)
        beta2  = theta[:, 1]   # (N,)
        rho    = theta[:, 2]   # (N,)
        zs     = theta[:, 3:]  # (N, T+1)

        # ── Hard support mask ────────────────────────────────
        out_of_support = (
            (sigma2 <= 0) | (beta2 <= 0) |
            (rho <= self.rho_lower) | (rho >= self.rho_upper)
        )

        # ── Log-prob accumulation ────────────────────────────
        lp  = scipy.stats.invgamma.logpdf(
                  sigma2, a=self.sigma2_alpha, scale=self.sigma2_beta)
        lp += scipy.stats.invgamma.logpdf(
                  beta2,  a=self.beta2_alpha,  scale=self.beta2_beta)
        lp += scipy.stats.uniform.logpdf(
                  rho, loc=self.rho_lower, scale=self.rho_upper - self.rho_lower)
        lp += scipy.stats.norm.logpdf(
                  zs[:, 0], loc=0, scale=sqrt(self.initial_distribution_variance))
        lp += np.sum(
            scipy.stats.norm.logpdf(
                zs[:, 1:],
                loc=rho[:, None] * zs[:, :-1],
                scale=np.sqrt(sigma2)[:, None],
            ),
            axis=1,
        )

        lp[out_of_support] = -np.inf
        return lp.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SIMULATOR  (unchanged — already correct and batched)
# ─────────────────────────────────────────────────────────────

def simulator(theta):
    """
    Batched simulator for the stochastic volatility model.
    
    Args:
        theta: (N, T+4) tensor or array of prior samples
               columns: [sigma2, beta2, rho, z_0, z_1, ..., z_T]
    
    Returns:
        xs: (N, T+1) tensor of simulated observations
    """
    # Convert to numpy if tensor
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()

    # Handle single sample case to ensure 2D
    if theta.ndim == 1:
        theta = theta[np.newaxis, :] # (1, T+4)

    beta2 = theta[:, 1] # (N,)
    beta = np.sqrt(beta2) # (N,)
    z = theta[:, 3:] # (N, T+1)

    # Scale matrix: beta_i * exp(x_{i,k} / 2) for all i, k
    scales = beta[:, None] * np.exp(z / 2) # (N, T+1)
    # Draw all observations at once
    xs = scipy.stats.norm.rvs(loc=0, scale=scales)  # (N, T+1)
    return torch.tensor(xs, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# PRIOR FACTORY — now calls process_prior
# ─────────────────────────────────────────────────────────────

def make_prior(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
               rho_lower, rho_upper, T, initial_distribution_variance=0.01):
    """
    Builds the raw SVMPrior, then wraps it with process_prior so that
    sbi gets a fully PyTorch-compatible Distribution with correct bounds.

    Bounds per dimension:
      [0]:  sigma2  in (0,   inf)
      [1]:  beta2   in (0,   inf)
      [2]:  rho     in (rho_lower, rho_upper)
      [3:]: z_k     in (-inf, inf)
    """
    raw_prior = SVMPrior(
        sigma2_alpha=sigma2_alpha, sigma2_beta=sigma2_beta,
        beta2_alpha=beta2_alpha,   beta2_beta=beta2_beta,
        rho_lower=rho_lower,       rho_upper=rho_upper,
        T=T,
        initial_distribution_variance=initial_distribution_variance,
    )

    # Build per-dimension bounds
    # sigma2, beta2 are positive  → lower=0, upper=+inf
    # rho is bounded               → lower=rho_lower, upper=rho_upper
    # z_0... z_T are real         → lower=-inf, upper=+inf
    lower_bound = torch.cat([
        torch.tensor([0.0, 0.0, rho_lower], dtype=torch.float32),  # sigma2, beta2, rho
        torch.full((T + 1,), -100,  dtype=torch.float32),  # z_0... z_T
    ])  # (T+4,)

    upper_bound = torch.cat([
        torch.tensor([1e6, 1e6, rho_upper], dtype=torch.float32),
        torch.full((T + 1,), 100, dtype=torch.float32),
    ])  # (T+4,)

    prior, theta_numel, prior_returns_numpy = process_prior(
        raw_prior,
        custom_prior_wrapper_kwargs={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    )

    print(f"  Prior processed successfully:")
    print(f"    theta_numel        = {theta_numel}")
    print(f"    prior_returns_numpy= {prior_returns_numpy}")

    return prior, theta_numel, prior_returns_numpy


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

path_to_repo      = Path(__file__).resolve().parents[4]
results_path      = str(path_to_repo / "results" / "real_examples" / "stoch_vol" / "npe_c")
trajectories_path = "/Users/Lieve/Documents/Masters Project/SBC-SBI/results/real_examples/stoch_vol/data/"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(x_observed_ID, num_sequential_rounds, num_simulations_per_round,
         use_combined_loss, density_estimator,
         sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
         rho_lower, rho_upper, T, initial_distribution_variance):

    # ── Load observation ─────────────────────────────────────
    trajectory = np.load(trajectories_path + f"trajectory{x_observed_ID}.npz")
    x_observed = trajectory["x"][:T + 1]
    x_obs_t    = torch.tensor(x_observed, dtype=torch.float32)

    # ── Build prior & simulator ──────────────────────────────
    prior, theta_numel, prior_returns_numpy = make_prior(
        sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta,
        rho_lower, rho_upper, T, initial_distribution_variance,
    )
    #diagnose_transform_many_samples(prior)   # ← add this

    # process_simulator handles:
    #   - numpy<->tensor casting  (because prior_returns_numpy=True)
    #   - batch looping           (our simulator is already batched, so no-op)
    #simulator_wrapped = process_simulator(simulator, prior, prior_returns_numpy)

    # ── Inference setup ──────────────────────────────────────
    inference = NPE_C(prior=prior, density_estimator=density_estimator)
    proposal  = prior

    posteriors_dict  = {}
    simulation_times = []
    training_times   = []

    for r in range(num_sequential_rounds):
        print(f"\n Round {r + 1} / {num_sequential_rounds}:")

        # ── Simulate ─────────────────────────────────────────
        t0 = time.perf_counter()
        # simulate_for_sbi uses the wrapped simulator and handles NaN rejection
        theta = proposal.sample((num_simulations_per_round,))
        x = simulator(theta)
        simulation_times.append(time.perf_counter() - t0)
        print(f"  Simulated {num_simulations_per_round} samples "
              f"in {simulation_times[-1]:.1f}s")

        # ── Train ─────────────────────────────────────────────
        t0 = time.perf_counter()
        inference.append_simulations(
            theta, x, proposal=proposal
        ).train(use_combined_loss=use_combined_loss)

        is_last_round = (r == num_sequential_rounds - 1)

        if is_last_round:
            # Don't fix x_obs on the final posterior — caller sets it
            sequential_posterior = inference.build_posterior()
        else:
            sequential_posterior = (
                inference.build_posterior().set_default_x(x_obs_t)
            )
            proposal = sequential_posterior

        training_times.append(time.perf_counter() - t0)
        posteriors_dict[f"round_{r}"] = sequential_posterior
        print(f"  Training done in {training_times[-1]:.1f}s")

    print("\n All rounds complete.")

    # ── Save ─────────────────────────────────────────────────
    config = {
        "x_observed_ID":                 x_observed_ID,
        "num_sequential_rounds":         num_sequential_rounds,
        "num_simulations_per_round":     num_simulations_per_round,
        "simulation_times":              simulation_times,
        "training_times":                training_times,
        "use_combined_loss":             use_combined_loss,
        "density_estimator":             density_estimator,
        "sigma2_alpha":                  sigma2_alpha,
        "sigma2_beta":                   sigma2_beta,
        "beta2_alpha":                   beta2_alpha,
        "beta2_beta":                    beta2_beta,
        "rho_lower":                     rho_lower,
        "rho_upper":                     rho_upper,
        "T":                             T,
        "initial_distribution_variance": initial_distribution_variance,
    }

    i = 0
    while os.path.exists(results_path + f"/sequential_posterior{i}.yaml"):
        i += 1

    config_path    = results_path + f"/sequential_posterior{i}.yaml"
    pkl_path       = results_path + f"/sequential_posterior{i}_posteriors_dict.pkl"

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    with open(pkl_path, "wb") as f:
        pickle.dump(posteriors_dict, f)

    print(f"\n Saved config     → {config_path}")
    print(f" Saved posteriors → {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_observed_ID",                type=int,   required=True)
    parser.add_argument("--num_sequential_rounds",         type=int,   default=4)
    parser.add_argument("--num_simulations_per_round",     type=int,   default=5000)
    parser.add_argument("--use_combined_loss",             action="store_true", default=False)
    parser.add_argument("--density_estimator",             type=str,   default="maf")
    parser.add_argument("--sigma2_alpha",                  type=float, default=3)
    parser.add_argument("--sigma2_beta",                   type=float, default=0.2)
    parser.add_argument("--beta2_alpha",                   type=float, default=3)
    parser.add_argument("--beta2_beta",                    type=float, default=5e-5)
    parser.add_argument("--rho_lower",                     type=float, default=-1.0)
    parser.add_argument("--rho_upper",                     type=float, default=1.0)
    parser.add_argument("--T",                             type=int,   required=True)
    parser.add_argument("--initial_distribution_variance", type=float, default=0.01)
    args = parser.parse_args()
    main(**vars(args))