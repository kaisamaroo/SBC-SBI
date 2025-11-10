import numpy as np
import torch
import matplotlib.pyplot as plt
from numbers import Number

def prior_pdf(theta):
    return np.exp(-theta**2/2)/np.sqrt(2*np.pi)

def likelihood_pdf(x, theta):
    return np.exp(-(x-theta)**2/2)/np.sqrt(2*np.pi)

def evidence_pdf(x):
    return np.exp(-x**2/4) / (2*np.sqrt(np.pi))

def posterior_pdf(theta, x):
    return np.exp(-(theta - x/2)**2) / (np.sqrt(np.pi))

def plot_approximate_posterior(approximate_posterior, prior_pdf, posterior_pdf, theta_range, x_observed):
    if isinstance(x_observed, Number):
        x_observed = torch.tensor([x_observed]) # Observed data
        fig, ax = plt.subplots(figsize=(10,5), dpi=150)

        # Plot prior
        label_prior = r"Prior $p(\theta) = N(\theta; 0,1)$"
        ax.plot(theta_range, prior_pdf(theta_range), label=label_prior, color="blue")

        # Plot true posterior
        label_posterior = fr"Posterior $p(\theta|{x_observed.item()}) = N(\theta; {x_observed.item()/2}, 1/2)$"
        ax.plot(theta_range, posterior_pdf(theta_range, x_observed), label=label_posterior, color="red", linestyle="--")

        # Plot 2 sigma confidence interval of prior
        ax.plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

        # Plot 2 sigma confidence interval of posterior
        ax.plot([(x_observed/2 - 2 * 0.5), (x_observed/2 + 2 * 0.5)], [0,0], color="red", linewidth=4, alpha=0.5)

        # Plot posterior approximation
        ax.plot(theta_range, approximate_posterior(theta_range, x_observed), color="green", label="Approximate Posterior")
        ax.set_title(f"Amortized (S)NPE-A with x_obs = {x_observed.item()}")

        ax.set_xlabel(r"$\theta$")
        ax.set_title(fr"Approximation to posterior")
    else:
        if len(x_observed)%2 != 0:
            raise ValueError("Must have even number of x_observed for plotting.")
        fig, ax = plt.subplots(nrows=int(len(x_observed)/2), ncols=2, figsize=(8, 8), dpi=150)
        for k, x_obs in enumerate(x_observed):
            i, j = k//2, k%2 # Coordinates for plotting

            # Plot prior
            label_prior = r"Prior $p(\theta) = N(0,1)$"
            ax[i,j].plot(theta_range, prior_pdf(theta_range), label=label_prior, color="blue")

            # Plot posterior
            label_posterior = fr"Posterior $p(\theta|{x_obs}) = N(\theta, {x_obs/2})$"
            ax[i,j].plot(theta_range, posterior_pdf(theta_range, x_obs), label=label_posterior, color="red", linestyle="--")

            # Plot 2 sigma confidence interval of prior
            ax[i,j].plot([-2,2], [0,0], color="blue", linewidth=4, alpha=0.5)

            # Plot 2 sigma confidence interval of posterior
            ax[i,j].plot([(x_obs/2 - 2 * 0.5), (x_obs/2 + 2 * 0.5)], [0,0], color="red", linewidth=4, alpha=0.5)

            # Plot posterior approximation
            ax[i,j].plot(theta_range, approximate_posterior(theta_range, x_obs), color="green", label="Approximate Posterior")
            ax[i,j].set_title(f"Posterior approximation with x_obs = {x_obs}")

            ax[i,j].set_xlabel(r"$\theta$")
            ax[i,j].set_title(fr"$p(\theta|{x_obs})$")
    plt.legend()
    plt.tight_layout()
    plt.show()
