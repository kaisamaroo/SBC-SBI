import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign # Kolmogorov distribution

def plot_sbc_ecdf(ranks, N_samp=250, N_iter=1000, alpha=0.05):
    ranks = np.array(ranks)
    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_title(f"{100*(1-alpha)}% credible interval for ECDF with" + r" $N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter}")
    plt.show()


def plot_sbc_ecdf_diff(ranks, N_samp=250, N_iter=1000, alpha=0.05):
    ranks = np.array(ranks)
    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_title(f"{100*(1-alpha)}% credible interval for" + r" ECDF($x$) - $x$ " + "with" + r" $N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter}")
    plt.show()