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

# FOR KALMAN FILTER:

def log_likelihood(x, rho, tau, sigma_true):
    """log p(x0:T|rho,tau)"""
    T = len(x) - 1
    sigma_true2 = sigma_true ** 2
    log_lik = 0.0
    for k in range(T + 1):
        if k == 0:
            mu_pred = 0.0
            s2_pred = 1.0
        else:
            mu_pred = rho * mu_filt
            s2_pred = rho ** 2 * s2_filt + tau ** 2
        s2_innov = s2_pred + sigma_true2
        innov = x[k] - mu_pred
        log_lik += scipy.stats.norm.logpdf(innov, loc=0.0, scale=np.sqrt(s2_innov))
        K = s2_pred / s2_innov
        mu_filt = mu_pred + K * innov
        s2_filt = s2_pred - K * s2_pred
    return log_lik


def log_prior(rho, tau, rho_lower, rho_upper,
              tau_loc, tau_scale, tau_lower, tau_upper):
    """log p(rho,tau)"""
    if not (rho_lower < rho < rho_upper):
        return -np.inf
    if not (tau_lower < tau < tau_upper):
        return -np.inf
    rho_logpdf = scipy.stats.uniform.logpdf(rho, loc=rho_lower, scale=rho_upper-rho_lower)
    tau_logpdf = scipy.stats.truncnorm.logpdf(tau,
                a=(tau_lower - tau_loc) / tau_scale,
                b=(tau_upper - tau_loc) / tau_scale,
                loc=tau_loc, scale=tau_scale)
    return rho_logpdf + tau_logpdf


def log_unnormalized_posterior(rho, tau, x, sigma_true,
                                rho_lower, rho_upper,
                                tau_loc, tau_scale, tau_lower, tau_upper):
    """log p(rho,tau) + log p(x0:T|rho,tau)"""
    return log_likelihood(x, rho, tau, sigma_true) + log_prior(
        rho, tau,
        rho_lower, rho_upper,
        tau_loc, tau_scale, tau_lower, tau_upper
    )


def proposal_rv(rho_old, tau_old, step_sizes):
    step_sizes = np.array(step_sizes)
    return np.array([rho_old, tau_old]) + step_sizes * np.random.randn(2)

  
def sample_z_given_theta_x(x, rho, tau, sigma):
    """sample from p(z0:T|x0:T,rho,tau)"""
    T = len(x) - 1
    sigma2 = sigma ** 2

    # Forward Kalman filter
    mu_filt = np.zeros(T + 1)
    s2_filt = np.zeros(T + 1)
    mu_pred = np.zeros(T + 1)
    s2_pred = np.zeros(T + 1)

    for k in range(T + 1):
        if k == 0:
            mp, sp = 0.0, 1.0
        else:
            mp = rho * mu_filt[k-1]
            sp = rho ** 2 * s2_filt[k-1] + tau ** 2
        mu_pred[k] = mp
        s2_pred[k] = sp
        s2_innov = sp + sigma2
        K = sp / s2_innov
        mu_filt[k] = mp + K * (x[k] - mp)
        s2_filt[k] = sp - K * sp

    # Backward sampler
    z = np.zeros(T + 1)
    z[T] = np.random.normal(mu_filt[T], np.sqrt(s2_filt[T]))
    for k in range(T - 1, -1, -1):
        G = s2_filt[k] * rho / s2_pred[k+1]
        mu_smooth = mu_filt[k] + G * (z[k+1] - mu_pred[k+1])
        s2_smooth = s2_filt[k] - G ** 2 * s2_pred[k+1]
        z[k] = np.random.normal(mu_smooth, np.sqrt(s2_smooth))
    return z


def metropolis_hastings(rho_0, tau_0, x, sigma_true, step_sizes, num_iterations,
                        rho_lower, rho_upper,
                        tau_loc, tau_scale, tau_lower, tau_upper):
    T = len(x) - 1
    rho_samples = np.zeros(num_iterations)
    tau_samples = np.zeros(num_iterations)

    z_samples = np.zeros((num_iterations, T+1))
    rho_prev = rho_0
    tau_prev = tau_0
    log_post_prev = log_unnormalized_posterior(
        rho_prev, tau_prev, x, sigma_true,
        rho_lower, rho_upper,
        tau_loc, tau_scale, tau_lower, tau_upper
    )
    rejections = 0
    for i in range(num_iterations):
        if i % (num_iterations // 10) == 0:
            print(f"{100 * i / num_iterations:.0f}% done")
        proposed = proposal_rv(rho_prev, tau_prev, step_sizes)
        rho_prop, tau_prop = proposed[0], proposed[1]
        log_post_prop = log_unnormalized_posterior(
            rho_prop, tau_prop, x, sigma_true,
            rho_lower, rho_upper,
            tau_loc, tau_scale, tau_lower, tau_upper
        )
        log_alpha = log_post_prop - log_post_prev
        if np.log(np.random.uniform()) < log_alpha:
            # Accepted
            rho_prev = rho_prop
            tau_prev = tau_prop
            log_post_prev = log_post_prop
        else:
            rejections += 1
        
        rho_samples[i] = rho_prev
        tau_samples[i] = tau_prev
        z_samples[i] = sample_z_given_theta_x(x, rho_prev, tau_prev, sigma_true)

    return rho_samples, tau_samples, z_samples, rejections / num_iterations


# FOR SBI:

def static_params_prior_rv(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper):
    # Prior for static parameters
    rho = scipy.stats.uniform.rvs(loc=rho_lower, scale=rho_upper - rho_lower)
    tau = scipy.stats.truncnorm.rvs(
                a=(tau_lower - tau_loc) / tau_scale,
                b=(tau_upper - tau_loc) / tau_scale,
                loc=tau_loc, scale=tau_scale
                )
    return rho, tau


def latent_trajectory_prior_rv(rho, tau, T):
    # Conditional prior for latent trajectory given static parameters
    zs = []
    for k in range(T+1):
        # Compute z_k
        if k==0:
            z = scipy.stats.norm.rvs(loc=0, scale=1)
        else:
            z = scipy.stats.norm.rvs(loc=rho * zs[k-1], scale=tau)
        # Append state to trajectory
        zs.append(z)
    return np.array(zs)


def prior_rv_sbi(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T):
    # Generate static parameters
    rho, tau = static_params_prior_rv(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper)
    # Generate latent trajectory given static parameters
    z = latent_trajectory_prior_rv(rho, tau, T)  # (T+1, )
    return np.concatenate([np.array([rho, tau]), z])  # (T+3, )


class SVMPrior:
    """
    Plain-class hierarchical prior for the lgss model.
    Implements.sample() and.log_prob() — the two methods process_prior
    requires. Returns numpy arrays from.sample() so that process_prior
    sets prior_returns_numpy=True and process_simulator handles the
    numpy<->tensor casting automatically.
    """

    def __init__(self, rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T):
        self.tau_loc = tau_loc
        self.tau_scale = tau_scale
        self.tau_lower = tau_lower
        self.tau_upper = tau_upper
        self.rho_lower = rho_lower
        self.rho_upper = rho_upper
        self.T = T

    def sample(self, sample_shape=torch.Size([])) -> np.ndarray:
        """
        Returns numpy array of shape (*sample_shape, T+3).
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
            prior_rv_sbi(self.rho_lower, self.rho_upper, self.tau_loc,
                         self.tau_scale, self.tau_lower, self.tau_upper, self.T
            )
            for _ in range(N)
        ]) # (N, T+3)

        if len(sample_shape) == 0:
            return theta.squeeze(0) # (T+3,)
        return theta.reshape(*sample_shape, self.T + 3)


    def log_prob(self, theta) -> torch.Tensor:
        """
        log p(theta) = log p(sigma2)
                     + log p(beta2)
                     + log p(rho)
                     + log p(z_0)
                     + sum_k log p(z_k | z_{k-1}, sigma2, rho)

        Args:
            theta: Tensor or ndarray of shape (N, T+3) or (T+3,)

        Returns:
            log_prob: Tensor of shape (N,)
        """
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        theta = np.atleast_2d(theta) # (N, T+3)

        if theta.shape[-1] != self.T + 3:
            return torch.full((theta.shape[0],), float('-inf'), dtype=torch.float32)

        rho = theta[:, 0] # (N,)
        tau = theta[:, 1] # (N,)
        zs = theta[:, 2:] # (N, T+1)

        # Support mask (lp is -inf outside of this)
        out_of_support = (
            (tau <= self.tau_lower) | (tau >= self.tau_upper) | (rho <= self.rho_lower) | (rho >= self.rho_upper)
        )

        # Log-prob accumulation
        lp = scipy.stats.truncnorm.logpdf(tau,
                a=(self.tau_lower - self.tau_loc) / self.tau_scale,
                b=(self.tau_upper - self.tau_loc) / self.tau_scale,
                loc=self.tau_loc, scale=self.tau_scale)
        lp += scipy.stats.uniform.logpdf(
                  rho, loc=self.rho_lower, scale=self.rho_upper - self.rho_lower)
        lp += scipy.stats.norm.logpdf(
                  zs[:, 0], loc=0, scale=1)
        lp += np.sum(
            scipy.stats.norm.logpdf(
                zs[:, 1:],
                loc=rho[:, None] * zs[:, :-1],
                scale=tau[:, None],
            ),
            axis=1,
        )

        lp[out_of_support] = -np.inf # Set log prob to -inf if the sample doesnt lie inside prior support
        return lp.astype(np.float32)
    

def make_prior(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T):
    """
    Builds the raw SVMPrior, then wraps it with process_prior so that
    sbi gets a fully PyTorch-compatible Distribution with correct bounds.

    Bounds per dimension:
      [0]:  rho     in (rho_lower, rho_upper)
      [1]:  tau     in (0, inf)
      [2:]: z_k     in (-inf, inf)
    """
    raw_prior = SVMPrior(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T)

    # Build per-dimension bounds
    # tau is positive
    # rho is bounded in an inverval
    # z_0... z_T are real but we need to impose constraints on these (so we use large numbers)
    lower_bound = torch.cat([
        torch.tensor([rho_lower, tau_lower], dtype=torch.float32), # rho, tau
        torch.full((T + 1,), -100,  dtype=torch.float32), # z_0... z_T
    ]) # (T+3,)

    upper_bound = torch.cat([
        torch.tensor([rho_upper, tau_upper], dtype=torch.float32),
        torch.full((T + 1,), 100, dtype=torch.float32),
    ]) # (T+3,)

    prior, _, __ = process_prior(
        raw_prior,
        custom_prior_wrapper_kwargs={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    )

    return prior


def simulator(theta, sigma_true):
    """
    Batched simulator for the lgss model.
    
    Args:
        theta: (N, T+3) tensor or array of prior samples
               columns: [rho, tau, z_0, z_1, ..., z_T]
    
    Returns:
        xs: (N, T+1) tensor of simulated observations
    """
    # Convert to numpy if tensor
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()

    # Handle single sample case to ensure 2D
    if theta.ndim == 1:
        theta = theta[np.newaxis, :] # (1, T+3)

    z = theta[:, 2:] # (N, T+1)

    xs = z + sigma_true * np.random.randn(*z.shape)
    return torch.tensor(xs, dtype=torch.float32)