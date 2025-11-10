"""
mog_1d_pdf

Implements density functions for a 1D Mixture of Gaussians (MOG) Bayesian model.

- Prior: zero-mean Gaussian with variance sigma^2.
- Likelihood: 1D MOG with variable mean shifts and unit variance.
- Posterior: resulting 1D MOG.
"""

from typing import List
import numpy as np


def prior_pdf(theta: float, sigma: float = 1.0) -> float:
    """
    Evaluate the PDF of a Gaussian prior N(0, sigma^2) at a given theta.

    Parameters
    ----------
    theta : float
        Value at which to evaluate the prior.
    sigma : float, optional
        Standard deviation of the prior (default 1.0).

    Returns
    -------
    float
        Prior PDF value at theta.
    """

    return np.exp(-theta**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def likelihood_pdf(x: float, theta: float, weights: List[float], mean_shifts: List[float]) -> float:
    """
    Evaluate the likelihood of a data point x under a 1D Mixture of Gaussians.
    
    Each component has variance 1 and mean given by theta - mean_shift.

    p(x | theta) = sum_k [ weights[k] * N(x; theta - mean_shifts[k], 1) ]

    Parameters
    ----------
    x : float
        Observation value.
    theta : float
        Parameter value on which to condition the likelihood.
    weights : List[float]
        Mixture weights for each Gaussian component.
    mean_shifts : List[float]
        Shifts for each component mean.

    Returns
    -------
    float
        Likelihood density evaluated at x.
    
    Raises
    ------
    ValueError
        If weights and mean_shifts have different lengths.
    """
    if len(weights) != len(mean_shifts):
        raise ValueError(
            f"weights and mean_shifts must have same length. "
            f"Got {len(weights)} and {len(mean_shifts)}."
        )

    density = 0.0
    for w, shift in zip(weights, mean_shifts):
        density += w * np.exp(-(x - theta - shift) ** 2 / 2) / np.sqrt(2 * np.pi)
    return density


def evidence_pdf(x: float, weights: List[float], mean_shifts: List[float], sigma: float = 1.0) -> float:
    """
    Compute the analytical evidence p(x) for the MOG Bayesian model.

    p(x) = sum_k [ w_k * N(x; mean_shift_k, 1 + sigma^2) ]

    Parameters
    ----------
    x : float
        Observation value.
    weights : List[float]
        Mixture weights.
    mean_shifts : List[float]
        Component mean shifts.
    sigma : float, optional
        Standard deviation of the prior (default 1.0).

    Returns
    -------
    float
        Evidence density at x.
    """
    return sum([w * np.exp(-(x-m_s)**2/(2*(1+sigma**2))) for w,m_s in zip(weights, mean_shifts)])/(np.sqrt(2*np.pi*(1+sigma**2)))


def posterior_pdf(theta: float, x: float, weights: List[float], mean_shifts: List[float], sigma: float = 1.0) -> float:
    """
    Compute the posterior PDF p(theta | x) for the MOG Bayesian model.

    Parameters
    ----------
    theta : float
        Latent variable.
    x : float
        Observation value.
    weights : List[float]
        Mixture weights.
    mean_shifts : List[float]
        Component mean shifts.
    sigma : float, optional
        Prior standard deviation (default 1.0).

    Returns
    -------
    float
        Posterior density at theta.
    """
    evidence_ = evidence_pdf(x, weights, mean_shifts, sigma)
    return (prior_pdf(theta, sigma) * likelihood_pdf(x, theta, weights, mean_shifts))/evidence_