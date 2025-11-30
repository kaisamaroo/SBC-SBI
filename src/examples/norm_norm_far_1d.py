import numpy as np
import torch
import matplotlib.pyplot as plt
from numbers import Number
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# Define simulator
def simulator(mu):
    return mu + torch.randn_like(mu)

# Define prior
prior = torch.distributions.MultivariateNormal(torch.tensor([0.]), torch.tensor([[1.]]))


def prior_pdf(theta):
    return np.exp(-theta**2/2)/np.sqrt(2*np.pi)


def likelihood_pdf(x, theta):
    return np.exp(-(x-theta)**2/2)/np.sqrt(2*np.pi)


def evidence_pdf(x):
    return np.exp(-x**2/4) / (2*np.sqrt(np.pi))


def posterior_pdf(theta, x):
    return np.exp(-(theta - x/2)**2) / (np.sqrt(np.pi))


def plot_approximate_posterior(approximate_posterior, prior_pdf, posterior_pdf, theta_range, x_observed, title=None):
    """
    For sbi posterior implementations
    
    approximate_posterior should be a function with 2 arguments (theta, x)

    x_observed should either be a Number or a list/array of Numbers

    theta_range should be 1D a torch tensor
    """
    if isinstance(x_observed, Number):
        x_observed = torch.tensor([x_observed]) # Observed data
        fig, ax = plt.subplots(figsize=(10,5), dpi=150)

        # Plot prior
        ax.plot(theta_range, prior_pdf(theta_range), label=f"prior with $2\sigma$ region", color="blue")

        # Plot true posterior
        ax.plot(theta_range, posterior_pdf(theta_range, x_observed), label="posterior with $2\sigma$ region", color="red", linestyle="--")

        # Plot 2 sigma confidence interval of prior
        ax.plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

        # Plot 2 sigma confidence interval of posterior
        ax.plot([(x_observed/2 - 2 * np.sqrt(0.5)), (x_observed/2 + 2 * np.sqrt(0.5))], [0,0], color="red", linewidth=4, alpha=0.5)

        # Plot posterior approximation
        ax.plot(theta_range, approximate_posterior(theta_range, x_observed), color="green", label="approximate posterior")
        ax.set_xlabel(r"$\theta$")
        if title:
            ax.set_title(title)
        else:
            ax.set_title(fr"Posterior approximation")
    else:
        if len(x_observed)%2 != 0:
            raise ValueError("Must have even number of x_observed for plotting.")
        fig, ax = plt.subplots(nrows=int(len(x_observed)/2), ncols=2, figsize=(8, 8), dpi=150)
        for k, x_obs in enumerate(x_observed):
            x_obs = torch.tensor([x_obs])
            i, j = k//2, k%2 # Coordinates for plotting

            # Plot prior
            ax[i,j].plot(theta_range, prior_pdf(theta_range), label=f"prior with $2\sigma$ region", color="blue")

            # Plot posterior
            ax[i,j].plot(theta_range, posterior_pdf(theta_range, x_obs), label=f"posterior with $2\sigma$ region", color="red", linestyle="--")

            # Plot 2 sigma confidence interval of prior
            ax[i,j].plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

            # Plot 2 sigma confidence interval of posterior
            ax[i,j].plot([(x_obs/2 - 2 * np.sqrt(0.5)), (x_obs/2 + 2 * np.sqrt(0.5))], [0,0], color="red", linewidth=4, alpha=0.5)

            # Plot posterior approximation
            ax[i,j].plot(theta_range, approximate_posterior(theta_range, x_obs), color="green", label="approximate posterior")
            ax[i,j].set_xlabel(r"$\theta$")
            ax[i,j].set_title(r"$x_{obs} =$" + f" {x_obs.item()}")
        if title:
            plt.suptitle(title)
        else:
            plt.suptitle(fr"Posterior approximation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_approximate_posterior_hist(approximate_posterior_samples, prior_pdf, posterior_pdf, theta_range, x_observed, title=None):
    """For sample-based posterior implementations
    
    Either approximate_posterior_samples is a list of Numbers or a list of lists/arrays

    Correspondingly, either x_observed is a Number of a list of Numbers
    """
    if isinstance(approximate_posterior_samples, list) and len(approximate_posterior_samples) > 0 and isinstance(approximate_posterior_samples[0], Number):
        fig, ax = plt.subplots(figsize=(10,5), dpi=150)

        # Plot prior
        ax.plot(theta_range, prior_pdf(theta_range), label=f"prior with $2\sigma$ region", color="blue")

        # Plot true posterior
        ax.plot(theta_range, posterior_pdf(theta_range, x_observed), label="posterior with $2\sigma$ region", color="red", linestyle="--")

        # Plot 2 sigma confidence interval of prior
        ax.plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

        # Plot 2 sigma confidence interval of posterior
        ax.plot([(x_observed/2 - 2 * np.sqrt(0.5)), (x_observed/2 + 2 * np.sqrt(0.5))], [0,0], color="red", linewidth=4, alpha=0.5)


        # Plot posterior approximation
        ax.hist(approximate_posterior_samples, color="green", density=True, bins=50, label="approximate posterior histogram")
        ax.set_xlabel(r"$\theta$")
        if title:
            ax.set_title(title)
        else:
            ax.set_title(fr"Posterior approximation")
    else:
        if len(x_observed)%2 != 0:
            raise ValueError("Must have even number of x_observed for plotting.")
        fig, ax = plt.subplots(nrows=int(len(x_observed)/2), ncols=2, figsize=(8, 8), dpi=150)
        for k, x_obs in enumerate(x_observed):
            i, j = k//2, k%2 # Coordinates for plotting

            # Plot prior
            ax[i,j].plot(theta_range, prior_pdf(theta_range), label=f"prior with $2\sigma$ region", color="blue")

            # Plot posterior
            ax[i,j].plot(theta_range, posterior_pdf(theta_range, x_obs), label=f"posterior with $2\sigma$ region", color="red", linestyle="--")

            # Plot 2 sigma confidence interval of prior
            ax[i,j].plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

            # Plot 2 sigma confidence interval of posterior
            ax[i,j].plot([(x_obs/2 - 2 * np.sqrt(0.5)), (x_obs/2 + 2 * np.sqrt(0.5))], [0,0], color="red", linewidth=4, alpha=0.5)

            # Plot posterior approximation
            ax[i,j].hist(approximate_posterior_samples[k], density=True, color="green", bins=50, label="approximate posterior histogram")
            ax[i,j].set_xlabel(r"$\theta$")
        if title:
            plt.suptitle(title)
        else:
            plt.suptitle(fr"Posterior approximation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def approximate_posterior_quantiles_against_x(approximate_posterior, x_range):
    """
    Compute the 0.5%, 12.5%, 50%, 87.5%, 99.5% quantiles of the approximate posterior given x
    for each x in x_range. Returns a list of 5 lists.
    """
    quantiles = [[],[],[],[],[]]
    print(len(quantiles))
    theta_range_for_interpolation = torch.linspace(-100,100,2000).view(-1,1)
    for x in x_range:
        pdf_vals = approximate_posterior(theta_range_for_interpolation, x * torch.ones_like(theta_range_for_interpolation))
        cdf_vals = cumulative_trapezoid(pdf_vals.view(-1), theta_range_for_interpolation.view(-1), initial=0)
        cdf_vals /= cdf_vals[-1]   # normalize just in case
        InvCDF = interp1d(cdf_vals, theta_range_for_interpolation.view(-1))
        for i, q in enumerate([InvCDF(0.005), InvCDF(0.125), InvCDF(0.5), InvCDF(0.875), InvCDF(0.995)]):
            quantiles[i].append(q)
    return quantiles


def plot_approximate_posterior_quantiles_against_x(x_range, quantiles, title=None):
    """
    Plot contour-style plot of x against the 0.5%, 12.5%, 50%, 87.5%, 99.5% quantiles 
    of the approximate posterior given x for each x in x_range.

    quantiles should be computed using approximate_posterior_quantiles_against_x
    """
    fig, ax = plt.subplots(figsize=(10,5))
    colors = ["red", "blue", "black", "blue", "red"]
    color_shaded_region = ["red", "blue", "red"]
    levels = [0.005, 0.125, 0.875, 0.995]
    labels = ["99% of approximate popsterior mass", "75% of approximate popsterior mass", "median of approximate popsterior", None, None]
    labels_shaded_region = ["99% of posterior mass", "75% of posterior mass", None]

    # Compute true posterior quantiles
    true_quantiles = [
        scipy.stats.norm.ppf(levels[i], loc=x_range/2, scale=np.sqrt(1/2))
        for i in range(len(levels))
    ]

    # Shade regions between successive true posterior quantiles
    for i in range(len(levels) - 1):
        ax.fill_between(
            x_range,
            true_quantiles[i],
            true_quantiles[i+1],
            color=color_shaded_region[i],
            alpha=0.3,
            label=labels_shaded_region[i]
        )

    # Plot your approximate posterior quantile curves
    for i in range(len(quantiles)):
        ax.plot(x_range, quantiles[i], color=colors[i], linewidth=2, label = labels[i])

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\theta$")

    plt.legend()
    plt.show()