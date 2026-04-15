import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from numbers import Number
from sbi.utils import process_prior
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from math import sqrt
import random
import pandas as pd

# flake8: noqa


def static_params_prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper):
    # Prior for static parameters
    sigma2 = scipy.stats.invgamma.rvs(a=sigma2_alpha, scale=sigma2_beta)
    beta2 = scipy.stats.invgamma.rvs(a=beta2_alpha,  scale=beta2_beta)
    rho = scipy.stats.uniform.rvs(loc=rho_lower, scale=rho_upper - rho_lower)
    return sigma2, beta2, rho


def latent_trajectory_prior_rv(sigma2, rho, T, initial_distribution_variance=0.01):
    # Conditional prior for latent trajectory given static parameters
    zs = []
    for k in range(T+1):
        # Compute z_k
        if k==0:
            z = scipy.stats.norm.rvs(loc=0, scale=sqrt(initial_distribution_variance))
        else:
            z = scipy.stats.norm.rvs(loc=rho * zs[k-1], scale=sqrt(sigma2))
        # Append state to trajectory
        zs.append(z)
    return np.array(zs)


def prior_rv_sbi(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower,
             rho_upper, T, initial_distribution_variance=0.01):
    # Generate static parameters
    sigma2, beta2, rho = static_params_prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha,
                                                beta2_beta, rho_lower, rho_upper)
    # Generate latent trajectory given static parameters
    z = latent_trajectory_prior_rv(sigma2, rho, T,
                                   initial_distribution_variance=initial_distribution_variance)  # (T+1, )
    return np.concatenate([np.array([sigma2, beta2, rho]), z])  # (T+4, )


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
        self.sigma2_beta = sigma2_beta
        self.beta2_alpha = beta2_alpha
        self.beta2_beta = beta2_beta
        self.rho_lower = rho_lower
        self.rho_upper = rho_upper
        self.T  = T
        self.initial_distribution_variance = initial_distribution_variance


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
            prior_rv_sbi(
                self.sigma2_alpha, self.sigma2_beta,
                self.beta2_alpha,  self.beta2_beta,
                self.rho_lower,    self.rho_upper,
                self.T,
                initial_distribution_variance=self.initial_distribution_variance,
            )
            for _ in range(N)
        ]) # (N, T+4)

        if len(sample_shape) == 0:
            return theta.squeeze(0) # (T+4,)
        return theta.reshape(*sample_shape, self.T + 4)


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
        theta = np.atleast_2d(theta) # (N, T+4)

        if theta.shape[-1] != self.T + 4:
            return torch.full((theta.shape[0],), float('-inf'), dtype=torch.float32)

        sigma2 = theta[:, 0] # (N,)
        beta2 = theta[:, 1] # (N,)
        rho = theta[:, 2] # (N,)
        zs = theta[:, 3:] # (N, T+1)

        # Support mask (lp is -inf outside of this)
        out_of_support = (
            (sigma2 <= 0) | (beta2 <= 0) |
            (rho <= self.rho_lower) | (rho >= self.rho_upper)
        )

        # Log-prob accumulation
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

        lp[out_of_support] = -np.inf # Set log prob to -inf if the sample doesnt lie inside prior support
        return lp.astype(np.float32)
    

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
        beta2_alpha=beta2_alpha, beta2_beta=beta2_beta,
        rho_lower=rho_lower, rho_upper=rho_upper,
        T=T,
        initial_distribution_variance=initial_distribution_variance,
    )

    # Build per-dimension bounds
    # sigma2, beta2 are positive
    # rho is bounded in an invercal
    # z_0... z_T are real but we need to impose constraints on these (so we use large numbers)
    lower_bound = torch.cat([
        torch.tensor([0.0, 0.0, rho_lower], dtype=torch.float32), # sigma2, beta2, rho
        torch.full((T + 1,), -100,  dtype=torch.float32), # z_0... z_T
    ]) # (T+4,)

    upper_bound = torch.cat([
        torch.tensor([1e6, 1e6, rho_upper], dtype=torch.float32),
        torch.full((T + 1,), 100, dtype=torch.float32),
    ]) # (T+4,)

    prior, _, __ = process_prior(
        raw_prior,
        custom_prior_wrapper_kwargs={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    )

    return prior


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


# For PMMH:

def f_rv(z_nm1, sigma, rho):
    return rho * z_nm1 + sigma * np.random.randn(len(z_nm1))

def f_pdf(z_n, z_nm1, sigma, rho):
    return scipy.stats.norm.pdf(z_n, loc=rho * z_nm1, scale=sigma)

def eta_rv(dim=1, initial_distribution_variance=0.01):
    return sqrt(initial_distribution_variance) * np.random.randn(dim)

def eta_pdf(z_0, initial_distribution_variance):
    return scipy.stats.norm.pdf(z_0, loc=0, scale=sqrt(initial_distribution_variance))

def q_rv(z_nm1, sigma, rho):
    return f_rv(z_nm1, sigma, rho)

def q_pdf(z_n, z_nm1, sigma, rho):
    return f_pdf(z_n, z_nm1, sigma, rho)

def q0_rv(dim=1, initial_distribution_variance=0.01):
    return eta_rv(dim, initial_distribution_variance)

def q0_pdf(z_0, initial_distribution_variance=0.01):
    return eta_pdf(z_0, initial_distribution_variance)

def g_rv(z_n, beta):
    return beta * np.exp(z_n / 2) * np.random.randn(len(z_n))

def g_pdf(x_n, z_n, beta):
    return scipy.stats.norm.pdf(x_n, loc=0, scale=beta * np.exp(z_n / 2))

def transition_matrix(Zkp1_samples, Zk_candidates, weights, N, M, k, sigma, rho):
    transition_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            Wk_j = weights[j, k]
            Zkp1_i = Zkp1_samples[i]
            Zk_j = Zk_candidates[j]
            unnormalized_transition_prob_i_to_j = Wk_j * f_pdf(Zkp1_i, Zk_j, sigma, rho)
            transition_matrix[i, j] = unnormalized_transition_prob_i_to_j
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    return transition_matrix


def SIR_loglikelihood(n, N, x, sigma, beta, rho, print_ks=False, return_FFBSa_sample=False, initial_distribution_variance=0.01):
    sample_paths_forwards = np.zeros((N, n+1))
    weights = np.zeros((N, n+1))
    points_grid = np.zeros((N, n+1))

    Z0_samples = q0_rv(dim=N, initial_distribution_variance=initial_distribution_variance)
    points_grid[:, 0] = Z0_samples
    log_w0 = scipy.stats.norm.logpdf(x[0], loc=0, scale=beta * np.exp(Z0_samples/2))
    logllk = scipy.special.logsumexp(log_w0) - np.log(N)

    w0 = np.exp(log_w0 - np.max(log_w0))
    w0_normalized = w0/np.sum(w0)
    weights[:, 0] = w0_normalized
    resample_idx = np.random.choice(N, size=N, p=w0_normalized)
    Z0_samples = Z0_samples[resample_idx]
    sample_paths_forwards[:,0] = Z0_samples

    for k in range(1, n+1):
        Zkm1_samples = sample_paths_forwards[:, k-1]
        Zk_samples = q_rv(Zkm1_samples, sigma=sigma, rho=rho)
        points_grid[:, k] = Zk_samples
        sample_paths_forwards[:,k] = Zk_samples
        log_wk = scipy.stats.norm.logpdf(x[k], loc=0, scale=beta * np.exp(Zk_samples/2))
        logllk += scipy.special.logsumexp(log_wk) - np.log(N)

        wk = np.exp(log_wk - np.max(log_wk))
        wk_normalized = wk/np.sum(wk)
        weights[:, k] = wk_normalized

        resample_idx = np.random.choice(N, size=N, p=wk_normalized)
        sample_paths_forwards = sample_paths_forwards[resample_idx]

    if not return_FFBSa_sample:
        return logllk

    M = 1

    sample_paths_backward = np.zeros((M, n+1))
    sample_idxs = np.zeros((M, n+1))

    probs_n = weights[:, n]
    Zn_samples_idx = np.random.choice(N, size=M, p=probs_n)
    sample_idxs[:, n] = Zn_samples_idx

    Zn_samples = points_grid[:,n][Zn_samples_idx]
    sample_paths_backward[:, n] = Zn_samples

    for k in reversed(range(0, n)):
        if print_ks:
            print(f"k = {k}")
        Zkp1_samples = sample_paths_backward[:, k+1]
        Zkp1_samples_idx = sample_idxs[:, k+1]
        Zk_candidates = points_grid[:, k]
        transition_matrix_k = transition_matrix(Zkp1_samples, Zk_candidates, weights, N, M, k, sigma, rho)
        cdf = np.cumsum(transition_matrix_k, axis=1)
        r = np.random.rand(len(Zkp1_samples_idx), 1)
        Zk_samples_idx = (cdf > r).argmax(axis=1)
        sample_idxs[:, k] = Zk_samples_idx
        Zk_samples = points_grid[:,k][Zk_samples_idx.astype(int)]
        sample_paths_backward[:, k] = Zk_samples

    return logllk, sample_paths_backward


def prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper):
    sigma2 = scipy.stats.invgamma.rvs(a=sigma2_alpha, scale=sigma2_beta)
    beta2 = scipy.stats.invgamma.rvs(a=beta2_alpha,  scale=beta2_beta)
    rho = scipy.stats.uniform.rvs(loc=rho_lower, scale=rho_upper - rho_lower)
    return np.array([sigma2, beta2, rho])


def prior_logpdf(sigma2, beta2, rho, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper):
    p_sigma2 = scipy.stats.invgamma.logpdf(sigma2, a=sigma2_alpha, scale=sigma2_beta)
    p_beta2  = scipy.stats.invgamma.logpdf(beta2,  a=beta2_alpha,  scale=beta2_beta)
    p_rho    = scipy.stats.uniform.logpdf(rho, loc=rho_lower, scale=rho_upper - rho_lower)
    return p_sigma2 + p_beta2 + p_rho


def prior_pdf(sigma2, beta2, rho, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper):
    p_sigma2 = scipy.stats.invgamma.pdf(sigma2, a=sigma2_alpha, scale=sigma2_beta)
    p_beta2  = scipy.stats.invgamma.pdf(beta2,  a=beta2_alpha,  scale=beta2_beta)
    p_rho    = scipy.stats.uniform.pdf(rho, loc=rho_lower, scale=rho_upper - rho_lower)
    return p_sigma2 * p_beta2 * p_rho


def proposal_rv(theta_old, step_sizes=(0.01, 5e-6, 0.01)):
    """Theta is a np array of length 3"""
    step_sizes = np.array(step_sizes)
    return theta_old + step_sizes * np.random.randn(3)


def proposal_logpdf(theta_new, theta_old, step_sizes=(0.01, 5e-6, 0.01)):
    """Thetas are np arrays of length 3"""
    step_sizes = np.array(step_sizes)
    return scipy.stats.multivariate_normal.logpdf(theta_new, mean=theta_old, cov=np.diag(step_sizes**2))


def PMMH_step_sample_trajectories(x, T, theta_km1, loglikelihood_km1, z_0_to_T_km1, N, sigma2_alpha,
                                  sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper,
                                  step_sizes=(0.01, 5e-6, 0.01), print_ks=False):
    theta_k = proposal_rv(theta_km1, step_sizes=step_sizes)
    sigma2_k, beta2_k, rho_k = theta_k
    if (sigma2_k <= 0) or (beta2_k <= 0) or (rho_k <= rho_lower) or (rho_k >= rho_upper):
        return theta_km1, loglikelihood_km1, z_0_to_T_km1, 0
    loglikelihood_k, z_0_to_T_k = SIR_loglikelihood(T, N, x, sqrt(sigma2_k), sqrt(beta2_k), rho_k, print_ks=print_ks, return_FFBSa_sample=True)
    sigma2_km1, beta2_km1, rho_km1 = theta_km1
    log_alpha = (
                loglikelihood_k + prior_logpdf(sigma2_k, beta2_k, rho_k,
                                               sigma2_alpha, sigma2_beta, beta2_alpha,
                                               beta2_beta, rho_lower, rho_upper)
              - loglikelihood_km1 - prior_logpdf(sigma2_km1, beta2_km1, rho_km1,
                                                 sigma2_alpha, sigma2_beta, beta2_alpha,
                                                 beta2_beta, rho_lower, rho_upper)
                )
    acceptance_prob = min(1, np.exp(log_alpha))
    if np.random.rand() < acceptance_prob:
        return theta_k, loglikelihood_k, z_0_to_T_k, acceptance_prob
    else:
        return theta_km1, loglikelihood_km1, z_0_to_T_km1, acceptance_prob


def PMMH_step(x, T, theta_km1, loglikelihood_km1, N, sigma2_alpha, sigma2_beta,
              beta2_alpha, beta2_beta, rho_lower, rho_upper, step_sizes=(0.01, 5e-6, 0.01)):
    theta_k = proposal_rv(theta_km1, step_sizes=step_sizes)
    sigma2_k, beta2_k, rho_k = theta_k
    if (sigma2_k <= 0) or (beta2_k <= 0) or (rho_k <= rho_lower) or (rho_k >= rho_upper):
        return theta_km1, loglikelihood_km1, 0
    loglikelihood_k = SIR_loglikelihood(T, N, x, sqrt(sigma2_k), sqrt(beta2_k), rho_k, return_FFBSa_sample=False)
    sigma2_km1, beta2_km1, rho_km1 = theta_km1
    log_alpha = (
                loglikelihood_k + prior_logpdf(sigma2_k, beta2_k, rho_k, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper)
              - loglikelihood_km1 - prior_logpdf(sigma2_km1, beta2_km1, rho_km1, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper)
                )
    acceptance_prob = min(1, np.exp(log_alpha))
    if np.random.rand() < acceptance_prob:
        return theta_k, loglikelihood_k, acceptance_prob
    else:
        return theta_km1, loglikelihood_km1, acceptance_prob


def generate_PMMH_samples(x, T, num_iterations, N, step_sizes, sigma2_alpha, sigma2_beta,
                          beta2_alpha, beta2_beta, rho_lower, rho_upper,
                          theta_0=None, sample_trajectories=False, print_ks=False):
    if theta_0 is None:
        theta_0 = prior_rv(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper)
    sigma2_0, beta2_0, rho_0 = theta_0
    if sample_trajectories:
        loglikelihood_0, z_0_to_T_0 = SIR_loglikelihood(T, N, x, sqrt(sigma2_0), sqrt(beta2_0), rho_0, print_ks=print_ks, return_FFBSa_sample=True)
    else:
        loglikelihood_0 = SIR_loglikelihood(T, N, x, sqrt(sigma2_0), sqrt(beta2_0), rho_0, print_ks=print_ks, return_FFBSa_sample=False)

    thetas = [theta_0]
    loglikelihoods = [loglikelihood_0]
    acceptance_ratios = [0]
    if sample_trajectories:
        z_0_to_Ts = [z_0_to_T_0]

    for k in range(1, num_iterations + 1):
        print(12*"-" + f"STEP {k}" + 12*"-")

        theta_km1 = thetas[k - 1]
        loglikelihood_km1 = loglikelihoods[k - 1]
        if sample_trajectories:
            z_0_to_T_km1 = z_0_to_Ts[k - 1]

        if sample_trajectories:
            theta_k, loglikelihood_k, z_0_to_T_k, acceptance_ratio_k = PMMH_step_sample_trajectories(x, T, theta_km1, loglikelihood_km1, z_0_to_T_km1, N, sigma2_alpha,
                                                                                                      sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper,
                                                                                                      step_sizes=step_sizes, print_ks=print_ks)
        else:
            theta_k, loglikelihood_k, acceptance_ratio_k = PMMH_step(x, T, theta_km1, loglikelihood_km1, N, sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower, rho_upper, step_sizes=step_sizes)

        thetas.append(theta_k)
        loglikelihoods.append(loglikelihood_k)
        acceptance_ratios.append(acceptance_ratio_k)
        if sample_trajectories:
            z_0_to_Ts.append(z_0_to_T_k)

        print(f"theta = {theta_k}")
        print(f"----- cum_logllk_var = {np.var(loglikelihoods)}")
        print(f"----- ar = {acceptance_ratio_k}")
        print(f"----- cum_mean_ar = {np.mean(acceptance_ratios)}")

    if sample_trajectories:
        return thetas, z_0_to_Ts, loglikelihoods, acceptance_ratios
    else:
        return thetas, loglikelihoods, acceptance_ratios