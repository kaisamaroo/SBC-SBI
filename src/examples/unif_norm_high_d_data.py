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
def simulator(mu, hadamard_matrix, sigma=1, d=1):
    # Hadamard matrix is k by k
    k, _ = hadamard_matrix.shape
    # Only works if we are up-projecting
    if k < d:
            raise AssertionError(f"k MUST BE LARGER THAN d! Currently k = {k} and d = {d}.")
    # Define rectangular Hadamard matrix
    H = hadamard_matrix[:, :d].to(dtype=mu.dtype)
    if mu.ndim > 1:
        batch_size, _ = mu.shape
        dist = MultivariateNormal(loc=mu,
                            covariance_matrix=sigma**2 * torch.eye(d))
        mu_plus_noise = dist.sample()
        x = mu_plus_noise @ H.T # (batch_size, d) @ (d, k) -> (batch_size, k)
        return x
    else:
        dist = MultivariateNormal(loc=mu,
                            covariance_matrix=sigma**2 * torch.eye(d))
        mu_plus_noise = dist.sample()
        x = H @ mu_plus_noise # (k, d) @ (d, ) -> (k, )
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


# Projection test function
def test_function_projection(theta, i):
    """
    theta is a 1D or 2D torch.tensor 

    project theta onto its i'th dimension
    """
    if theta.dim() == 1:
        return theta[i]
    else:
        # In this case, theta is (batch_size, parameter_dimension)
        return theta[:, i]


def test_function_squared_norm(theta):
    """
    theta is a 1D or 2D torch.tensor 

    return squared norm of parameter vector
    """
    if theta.dim() == 1:
        return torch.sum(theta**2)
    else:
        # In this case, theta is (batch_size, parameter_dimension)
        return torch.sum(theta**2, axis=1)


# UPDATE THIS ALONG WITH all_test_function_names
def get_test_function(test_function_name="projection0", d=1):
    if not test_function_name:
        # If None is input, return None
        return None
    for i in range(d):
        if test_function_name == f"projection{i}":
            return lambda x : test_function_projection(x, i)
    if test_function_name=="squared_norm":
        return test_function_squared_norm
    else:
        raise NotImplementedError("No matching test function.")
    
# UPDATE THIS ALONG WITH get_test_function
def get_all_test_function_names_list(d=1):
    all_test_function_names = [f"projection{i}" for i in range(d)] \
                        + ["squared_norm"]
    return all_test_function_names


def plot_leakage_factors(leakage_factor_dict, type, alpha=0.1, ax=None, logscale=False):
    # TO DO: Implement a heatmap for nr_nspr 
    user_defined_ax = ax
    if not ax:
        fig, ax = plt.subplots(figsize=(10,5))

    allowed_types = ["d", "ns", "nr_nspr"]
    if not type in allowed_types:
        raise NotImplementedError(f"TYPE {type} NOT RECOGNISED! MUST BE ONE OF: 'd', 'ns', 'ns_nspr'!")

    if type in ["d", "ns"]:
        # Experiments that vary only 1 parameter (d or ns)
        x = [k for k in leakage_factor_dict.keys()]
    else:
        # Experiments that vary nr_nspr
        # Make axis integer labels and then re-label the x-axis with tuples
        x = range(len(leakage_factor_dict))

    median_leakage_factor_list = []
    lq_leakage_factor_list = []
    uq_leakage_factor_list = []

    for k, v in leakage_factor_dict.items():
        v = v[~np.isnan(v)] # Remove nans (caused when SNPE-C failed)
        median = np.quantile(v, 0.5)
        uq = np.quantile(v, 1 - alpha / 2)
        lq = np.quantile(v, alpha / 2)
        median_leakage_factor_list.append(median)
        lq_leakage_factor_list.append(lq)
        uq_leakage_factor_list.append(uq)
    
    ax.plot(x, median_leakage_factor_list, label="median")
    ax.fill_between(x, lq_leakage_factor_list, uq_leakage_factor_list, alpha=0.3, label=f"{100*(alpha / 2)}% and {100*(1 - alpha / 2)}% quantiles")
    
    if logscale:
        ax.set_xscale('log')
    ax.set_xlabel(type)
    ax.set_ylabel("Leakage factor")
    ax.set_title("Leakage factors against " + type)
    if type == "nr_nspr":
        ax.set_xticks(x, [str(t) for t in leakage_factor_dict.keys()])

    plt.legend()
    plt.show()












