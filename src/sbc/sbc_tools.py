import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign # Kolmogorov distribution
import torch

def plot_sbc_histogram(ranks, N_iter=1000, N_samp=250, title=None, ax=None):
    ranks = np.array(ranks)

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    # Plot Histogram
    ax.hist(ranks, density=True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"SBC histogram " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()

    return ax

def plot_sbc_ecdf(ranks, N_iter=1000, N_samp=250, alpha=0.05, title=None, ax=None):
    ranks = np.array(ranks)

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    unit_range = np.linspace(0, 1, 1000)
    x_grid = np.arange(0, N_samp + 1) / N_samp

    k_alpha = kstwobign.ppf(1-alpha) # (1-alpha) quantile of Kolmogorov distribution

    # Plot approximate asymptotic 1-alpha simultaneous credible region for ECDF
    ax.fill_between(unit_range,
        unit_range - k_alpha / np.sqrt(N_iter),
        unit_range + k_alpha / np.sqrt(N_iter),
        alpha=0.3)
    
    ECDF = np.vectorize(lambda x : np.sum(ranks <= x) / N_iter)

    # Plot ECDF
    ax.plot(x_grid, ECDF(x_grid))

    ax.set_xlabel("x")
    ax.set_ylabel("ECDF")
    ax.set_ylim((0,1))
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Sample ECDF with approximate {100*(1-alpha)}% interval " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()

    return ax

def plot_sbc_ecdf_diff(ranks, N_iter=1000, N_samp=250, alpha=0.05, title=None, ax=None):
    ranks = np.array(ranks)

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    unit_range = np.linspace(0, 1, 1000)
    x_grid = np.arange(0, N_samp + 1) / N_samp

    k_alpha = kstwobign.ppf(1-alpha) # (1-alpha) quantile of Kolmogorov distribution

    # Plot approximate asymptotic 1-alpha simultaneous credible region for ECDF
    ax.fill_between(unit_range,
        -1 * k_alpha / np.sqrt(N_iter),
        k_alpha / np.sqrt(N_iter),
        alpha=0.3)
    
    ECDF = np.vectorize(lambda x : np.sum(ranks <= x) / N_iter)

    # Plot ECDF
    ax.plot(x_grid, ECDF(x_grid) - x_grid)

    ax.set_xlabel("x")
    ax.set_ylabel(r"ECDF($x$) - $x$")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(r"ECDF($x$) - $x$ " + f"with approximate {100*(1-alpha)}% interval " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()
    return ax

def plot_sbc_all(ranks, N_iter=1000, N_samp=250, alpha=0.05, title=None):
    ranks = np.array(ranks)

    fig, ax = plt.subplots(figsize=(10, 5), ncols=3)

    plot_sbc_histogram(ranks, N_iter=N_iter, N_samp=N_samp, title="Histogram", ax=ax[0])
    plot_sbc_ecdf(ranks, N_iter=N_iter, N_samp=N_samp, alpha=alpha, title=f"ECDF with {100*(1-alpha)}% interval", ax=ax[1])
    plot_sbc_ecdf_diff(ranks, N_iter=N_iter, N_samp=N_samp, alpha=alpha, title=f"ECDF - x with {100*(1-alpha)}% interval", ax=ax[2])

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle("SBC diagnostic plots " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")

    plt.tight_layout()
    plt.show()
    return ax

def sbc_ranks(model, prior, posterior, test_function=None, N_iter=100, N_samp=100):
    """
    return normalized SBC ranks
    """
    ranks = []
    for i in range(N_iter):
        prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
        simulated_datapoint = model(prior_sample) # Simulate a datapoint from the model given the prior sample. Returns 1d tensor
        posterior_samples = posterior.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
        if test_function:
            rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
        else:
            # If no test function provided, assume theta is 1D.
            rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
        ranks.append(float(rank))
    return ranks


def sbc_ranks_snpe_a(simulator,
                         prior,
                         train_sequential_posterior,
                         test_function=None,
                         N_iter=100,
                         N_samp=100,
                         num_sequential_rounds=4,
                         num_simulations_per_round=5000,
                         show_progress=True):
    """
    return normalized SBC ranks for SNPE-A, being careful to account for errors
    due to non-spd covariance matrices
    """
    failed_round_counter = 0
    ranks = []
    for i in range(N_iter):
        if show_progress:
            print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
        prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
        simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
        try:
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round)
            try:
                posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
                if test_function:
                    rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
                else:
                    # If no test function provided, assume theta is 1D.
                    rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
                ranks.append(float(rank))
            except AssertionError:
                print(f"\n SBC ROUND {i+1} FINAL POSTERIOR NOT SPD! SKIPPING ROUND")
                ranks.append(np.nan)
                failed_round_counter += 1
        except AssertionError:
            print(f"\n SBC ROUND {i+1} HAD A PROPOSAL PRIOR NOT SPD! SKIPPING ROUND")
            ranks.append(np.nan)
            failed_round_counter += 1
    if show_progress:
        print("\n" + 12*"-" + f"FINISHED SBC WITH {failed_round_counter} FAILED RUNS" + 12*"-")
    return ranks
