import numpy as np
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from numbers import Number
from sbi.utils import BoxUniform
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# Define simulator
def simulator(mu, sigma=1, d=1):
    dist = MultivariateNormal(loc=mu,
                           covariance_matrix=sigma**2 * torch.eye(d))
    x = dist.sample() # shape(batch_size, d)
    return x


def make_prior(L=-1, U=1, d=1):
    prior = BoxUniform(
        torch.tensor([L for _ in range(d)]),
        torch.tensor([U for _ in range(d)])
    )
    return prior


def prior_pdf(mu, L=-1, U=1, d=1):
    if torch.all((mu >= L) and (mu <= U)):
        return 1/2**d
    else:
        return 0
    

def true_posterior_pdf(mu, x, L=-1.0, U=1.0, sigma=1.0):
    a = (L - x) / sigma
    b = (U - x) / sigma
    return scipy.stats.truncnorm.pdf(mu, a, b, loc=x, scale=sigma)


def get_approximate_posterior_density(posterior):
    """
    Return the pdf of the approximate posterior defined by posterior

    posterior should be a DirectPosterior object
    """
    def approximate_posterior_amortized(theta, x):
        """
        Output SNPE-A analytical posterior approximation \hat{p}(theta | x)

        theta and x must be 1D torch tensors
        """
        return torch.exp(posterior.potential(theta, x))
    return approximate_posterior_amortized


def approximate_posterior_quantiles_against_x(posterior, x_range, num_samples=5000):
    """
    ONLY WORKS FOR POSTERIORS IMPLEMENTED WITH d=1
    Compute 0.5%, 12.5%, 50%, 87.5%, 99.5% quantiles of a sampler-based approximate posterior 
    for each x in x_range. 
    
    Returns a list of 5 lists: one list per quantile level.
    """
    test_sample = posterior.sample((1, ), x=x_range[0], show_progress_bars=False).view(-1)
    if len(test_sample) > 1:
        raise ValueError(f"THIS FUNCTION ONLY WORKS FOR ONE-DIMENSIONAL POSTERIORS (d = 1)! Currently, d = {len(test_sample)}")

    quantiles = [[],[],[],[],[]]
    for x in x_range:
        # draw samples
        samples = posterior.sample((num_samples, ), x=x, show_progress_bars=False)
        # convert to numpy for quantile computation
        samples_np = samples.view(-1,1).numpy()
        # compute requested quantiles
        quantile_vals = np.quantile(samples_np, [0.005, 0.125, 0.5, 0.875, 0.995])
        for i, q in enumerate(quantile_vals):
            quantiles[i].append(q)
    return quantiles


def plot_approximate_posterior_quantiles_against_x(x_range, quantiles, sigma, L, U, title=None, linewidth=1, ax=None, samples=None, s=1):
    """
    Plot contour-style plot of x against the 0.5%, 12.5%, 50%, 87.5%, 99.5% quantiles 
    of the approximate posterior given x for each x in x_range.

    quantiles should be computed using approximate_posterior_quantiles_against_x
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))
    colors = ["red", "blue", "black", "blue", "red"]
    color_shaded_region = ["red", "blue", "red"]
    levels = [0.005, 0.125, 0.875, 0.995]
    labels = ["99% of approximate posterior mass", "75% of approximate posterior mass", "median of approximate posterior", None, None]
    labels_shaded_region = ["99% of posterior mass", "75% of posterior mass", None]

    # If samples are given, plot them
    if samples:
        # Samples is either a dict of np arrays, or a dict of lists of np arrays
        if isinstance(samples["parameter_samples"], list):
            # Multi-round samples (sequential model)
            for r in range(len(samples["parameter_samples"])):
                parameter_samples = samples["parameter_samples"][r]
                data_samples = samples["data_samples"][r]
                # Plot samples from round r
                if r == len(samples["parameter_samples"]) - 1:
                    # Plot final round samples black
                    ax.scatter(data_samples, parameter_samples, label="Training data samples (final round)", alpha=0.6, s=s, color="black")
                else:
                    ax.scatter(data_samples, parameter_samples, label=f"Training data samples (round {r+1})", alpha=0.6, s=s)
        else:
            # Single-round samples (amortized model)
            parameter_samples = samples["parameter_samples"]
            data_samples = samples["data_samples"]
            ax.scatter(data_samples, parameter_samples, label="Training data samples", alpha=0.6, s=s, color="black")

    # Compute true posterior quantiles
    a = (L - x_range) / sigma
    b = (U - x_range) / sigma
    true_quantiles = [
        scipy.stats.truncnorm.ppf(levels[i], a, b, loc=x_range, scale=sigma)
        for i in range(len(levels))
    ]

    true_median = scipy.stats.truncnorm.ppf(0.5, a, b, loc=x_range, scale=sigma)
    ax.plot(x_range, true_median, color="grey", linestyle="--", label="median of true posterior")

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
        ax.plot(x_range, quantiles[i], color=colors[i], linewidth=linewidth, label = labels[i])

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\theta$")
    ax.set_xlim((np.min(x_range), np.max(x_range)))
    ax.legend()
    if not ax:
        plt.legend()
        plt.show()
    return ax


def plot_approximate_posterior_quantiles_diff_against_x(x_range, quantiles, sigma, L, U, title=None, linewidth=1, ax=None, samples=None, s=1):
    """
    Plot contour-style plot of x against the 0.5%, 12.5%, 50%, 87.5%, 99.5% quantiles 
    of the approximate posterior given x for each x in x_range.

    quantiles should be computed using approximate_posterior_quantiles_against_x
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))
    colors = ["red", "blue", "black", "blue", "red"]
    color_shaded_region = ["red", "blue", "red"]
    levels = [0.005, 0.125, 0.875, 0.995]
    labels = ["99% of approximate posterior mass", "75% of approximate posterior mass", "median of approximate posterior", None, None]
    labels_shaded_region = ["99% of posterior mass", "75% of posterior mass", None]

    # If samples are given, plot them
    if samples:
        # Samples is either a dict of np arrays, or a dict of lists of np arrays
        if isinstance(samples["parameter_samples"], list):
            # Multi-round samples (sequential model)
            for r in range(len(samples["parameter_samples"])):
                parameter_samples = samples["parameter_samples"][r]
                data_samples = samples["data_samples"][r]
                post_mean_on_samples = (sigma**2 / (1 + sigma**2)) * data_samples
                # Plot samples from round r
                if r == len(samples["parameter_samples"]) - 1:
                    # Plot final round samples black
                    ax.scatter(data_samples, parameter_samples - post_mean_on_samples, label="Training data samples (final round)", alpha=0.6, s=s, color="black")
                else:
                    ax.scatter(data_samples, parameter_samples - post_mean_on_samples, label=f"Training data samples (round {r+1})", alpha=0.6, s=s)
        else:
            # Single-round samples (amortized model)
            parameter_samples = samples["parameter_samples"]
            data_samples = samples["data_samples"]
            post_mean_on_samples = (sigma**2 / (1 + sigma**2)) * data_samples
            ax.scatter(data_samples, parameter_samples - post_mean_on_samples, label="Training data samples", alpha=0.6, s=s, color="black")
    
    # Compute true posterior quantiles
    a = (L - x_range) / sigma
    b = (U - x_range) / sigma
    true_quantiles = [
        scipy.stats.truncnorm.ppf(levels[i], a, b, loc=x_range, scale=sigma)
        for i in range(len(levels))
    ]

    true_median = scipy.stats.truncnorm.ppf(0.5, a, b, loc=x_range, scale=sigma)
    ax.plot(x_range, np.zeros(len(x_range)), color="grey", linestyle="--", label="median of posterior")

    # Shade regions between successive true posterior quantiles
    for i in range(len(levels) - 1):
        ax.fill_between(
            x_range,
            true_quantiles[i] - true_median,
            true_quantiles[i+1] - true_median,
            color=color_shaded_region[i],
            alpha=0.3,
            label=labels_shaded_region[i]
        )

    # Plot your approximate posterior quantile curves
    for i in range(len(quantiles)):
        ax.plot(x_range, quantiles[i] - true_median, color=colors[i], linewidth=linewidth, label = labels[i])

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\theta$")
    ax.set_xlim((np.min(x_range), np.max(x_range)))
    ax.legend()
    if not ax:
        plt.legend()
        plt.show()
    return ax


def snpe_a_posterior_variance(x, sequential_posterior):
    """
    Return variance of posterior approximation for various possible observed x
    
    Note that this is the variance after the final proposal correction, and therefore may
    become negative (well known issue with SNPE-A).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([float(x)])
    x = x.view(-1,1)

    # By default, sbi z-scores theta using an approximation to the prior variance
    # We obtain this estimator sigma_hat below so that we can undo this transformation
    sigma_hat = 1 / sequential_posterior.posterior_estimator._neural_net.net._transform._transforms[0]._scale
    # Precision of proposal prior in z-scored theta space
    latent_prec_proposal_prior = sequential_posterior.posterior_estimator._prec_pp.squeeze() # Precision of proposal prior
    # Obtain precision of the (uncorrected) MDN
    embedded_x = sequential_posterior.posterior_estimator._neural_net.net._embedding_net(x)
    dist = sequential_posterior.posterior_estimator._neural_net.net._distribution
    _, _, latent_prec_MDN, _, _ = dist.get_mixture_components(embedded_x) # Precision of density estimator
    latent_prec_MDN = latent_prec_MDN.squeeze()
    # Re-scale precisions using sigma_hat to undo z-scoring transformation
    var_MDN = sigma_hat ** 2 / latent_prec_MDN
    var_proposal_prior = sigma_hat ** 2 / latent_prec_proposal_prior
    return 1 / ((1 / var_MDN) - (1 / var_proposal_prior))


def plot_snpe_a_posterior_variance(x_range, sequential_posterior, ylim=None, title=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))
    x_range = torch.tensor(x_range)
    posterior_variances = snpe_a_posterior_variance(x_range, sequential_posterior).detach()
    ax.plot(x_range, np.where(posterior_variances >= 0, posterior_variances, np.nan), color="green", label="Positive posterior MDN variance (posterior well defined)")
    ax.plot(x_range, np.where(posterior_variances < 0, posterior_variances, np.nan), color="red", label="Negative posterior MDN variance (posterior not defined)")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("Variance")
    if not title:
        title = r"Variance of SNPE-A posterior approximation $\tilde{\pi}(\theta | x)$ for different x."
    ax.set_title(title)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if not ax:
        plt.show()
    return ax


def plot_sequential_samples(sequential_posterior_simulations, sequential_posterior_config, x_observed, sigma, plot_true_posterior=False, axis_limits = None, alpha=0.5, title=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))
    num_sequential_rounds = sequential_posterior_config["num_sequential_rounds"]
    num_simulations_per_round = sequential_posterior_config["num_simulations_per_round"]
    for r in range(num_sequential_rounds):
        parameter_samples = sequential_posterior_simulations[f"parameter_samples_round_{r}"]
        data_samples = sequential_posterior_simulations[f"data_samples_round_{r}"]
        ax.scatter(data_samples, parameter_samples, alpha=alpha, label=f"Round {r+1} samples" + (" (final round)" if r+1==num_sequential_rounds else ""))
    true_posterior_param_samples = true_posterior(sigma, x_observed).sample((num_simulations_per_round,))
    true_posterior_data_samples = simulator(true_posterior_param_samples)
    ax.scatter(true_posterior_data_samples, true_posterior_param_samples, alpha=alpha, label="True posterior samples")
    if plot_true_posterior:
        ax.axhline(x_observed*(sigma**2)/(1+sigma**2), color="k", label="True posterior mean", linestyle="--")
        ax.axhline(x_observed*(sigma**2)/(1+sigma**2) - 2*(sigma**2)/(1+sigma**2), color="blue", label="95% region", linestyle="--")
        ax.axhline(x_observed*(sigma**2)/(1+sigma**2) + 2*(sigma**2)/(1+sigma**2), color="blue", linestyle="--")
        ax.axhline(x_observed*(sigma**2)/(1+sigma**2) - 3*(sigma**2)/(1+sigma**2), color="red", label="99% region", linestyle="--")
        ax.axhline(x_observed*(sigma**2)/(1+sigma**2) + 3*(sigma**2)/(1+sigma**2), color="red", linestyle="--")
        ax.axvline(x_observed, color="k", linestyle="--")
    if axis_limits:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
    if not title:
        title = r"Training samples for each round of SNPE."
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\theta$")
    if not ax:
        plt.show()
    return ax