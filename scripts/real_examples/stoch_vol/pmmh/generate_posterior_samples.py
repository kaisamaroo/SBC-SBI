# flake8: noqa

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.tsaplots import plot_acf, acf
from matplotlib.ticker import PercentFormatter
from scipy.integrate import quad
from math import cos, sqrt
import pandas as pd
import torch
import numpy as np
from examples.stoch_vol import generate_PMMH_samples
import pickle
import argparse
from pathlib import Path
import os
import yaml
import time
SEED = 49
random.seed(SEED)
np.random.seed(SEED)

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "stoch_vol" / "pmmh")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "stoch_vol" / "data")


def main(x_observed_ID, num_iterations, N, step_sizes, theta_0, sample_trajectories, print_ks,
         sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower,
         rho_upper, T, initial_distribution_variance):
    
    # Convert string arguments to bool
    if sample_trajectories in ["True", "true"]:
        sample_trajectories = True
    else:
        sample_trajectories = False

    # Convert string arguments to bool
    if print_ks in ["True", "true"]:
        print_ks = True
    else:
        print_ks = False

    # Import data to condition on
    trajectory = np.load(trajectories_path + f"/trajectory{x_observed_ID}.npz")
    x_observed = trajectory["x"]
    x_observed = x_observed[:T+1] # Only take first T+1 samples from trajectory

    print("\n STARTING PMMH:")
    start_time = time.perf_counter()
    if sample_trajectories:
        thetas, z_0_to_Ts, loglikelihoods, acceptance_ratios = generate_PMMH_samples(x_observed, T,
                                                                        num_iterations=num_iterations,
                                                                        N=N,
                                                                        step_sizes=step_sizes,
                                                                        sigma2_alpha=sigma2_alpha,
                                                                        sigma2_beta=sigma2_beta,
                                                                        beta2_alpha=beta2_alpha,
                                                                        beta2_beta=beta2_beta,
                                                                        rho_lower=rho_lower,
                                                                        rho_upper=rho_upper,
                                                                        theta_0=theta_0,
                                                                        sample_trajectories=sample_trajectories,
                                                                        print_ks=print_ks)
        z_0_to_Ts = np.array(z_0_to_Ts).squeeze(1) # Shape (num_iterations+1, T+1)
    else:
        thetas, loglikelihoods, acceptance_ratios = generate_PMMH_samples(x_observed, T,
                                                                        num_iterations=num_iterations,
                                                                        N=N,
                                                                        step_sizes=step_sizes,
                                                                        sigma2_alpha=sigma2_alpha,
                                                                        sigma2_beta=sigma2_beta,
                                                                        beta2_alpha=beta2_alpha,
                                                                        beta2_beta=beta2_beta,
                                                                        rho_lower=rho_lower,
                                                                        rho_upper=rho_upper,
                                                                        theta_0=theta_0,
                                                                        sample_trajectories=sample_trajectories,
                                                                        print_ks=print_ks)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("\n PMMH FINISHED.")

    posterior_samples_dict = {}
    posterior_samples_dict["thetas"] = thetas
    posterior_samples_dict["loglikelihoods"] = loglikelihoods
    posterior_samples_dict["acceptance_ratios"] = acceptance_ratios
    if sample_trajectories:
        posterior_samples_dict["z_0_to_Ts"] = z_0_to_Ts

    config = {"x_observed_ID": x_observed_ID,
              "total_time": total_time,
              "sigma2_alpha": sigma2_alpha,
              "sigma2_beta": sigma2_beta,
              "beta2_alpha": beta2_alpha,
              "beta2_beta": beta2_beta,
              "rho_lower": rho_lower,
              "rho_upper": rho_upper,
              "T": T,
              "initial_distribution_variance": initial_distribution_variance,
              "num_iterations": num_iterations,
              "N": N,
              "step_sizes": step_sizes,
              "theta_0": theta_0,
              "sample_trajectories": sample_trajectories,
              "print_ks": print_ks}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/posterior_samples{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + f"/posterior_samples{i}.yaml"
    posterior_samples_save_path = results_path + f"/posterior_samples{i}.npz"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving posterior samples to {posterior_samples_save_path}:")
    np.savez(posterior_samples_save_path, **posterior_samples_dict)
    print(f"\n Posterior samples saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_observed_ID", type=int, required=True)

    parser.add_argument("--num_iterations", type=int, default=1000,
                            help="Number of PMMH iterations (not including initial iterate)")
    parser.add_argument("--step_sizes", nargs="+", type=float, default=[0.03, 4e-6, 0.03], # These are very rough defaults
                            help="Step sizes for sigma^2, beta^2, rho in PMMH")
    parser.add_argument("--N", type=int, default=1000,
                            help="Number of SIR samples to approximate the likelihood at each PMMH iteration")
    parser.add_argument("--theta_0", nargs="+", type=float, default=None,
                            help="Initial parameter iterate for PMMH. If None, it initializes using a prior sample")
    parser.add_argument("--sample_trajectories", type=str, default="True",
                            help="Whether to also sample latent trajectories z0:T or just static parameters")
    parser.add_argument("--print_ks", type=str, default="False",
                            help="Whether to print backward progress during FFBSa")

    parser.add_argument("--sigma2_alpha", type=float, default=3,
                        help="Shape parameter alpha for InvGamma prior on sigma^2")
    parser.add_argument("--sigma2_beta", type=float, default=0.2,
                        help="Scale parameter beta for InvGamma prior on sigma^2")
    parser.add_argument("--beta2_alpha", type=float, default=3,
                        help="Shape parameter alpha for InvGamma prior on beta^2")
    parser.add_argument("--beta2_beta", type=float, default=5e-5,
                        help="Scale parameter beta for InvGamma prior on beta^2")
    parser.add_argument("--rho_lower", type=float, default=0.0,
                        help="Lower bound of Uniform prior on rho")
    parser.add_argument("--rho_upper", type=float, default=1.0,
                        help="Upper bound of Uniform prior on rho")

    parser.add_argument("--T", type=int, required=True,
                        help="Length of observed time series (number of steps)")
    parser.add_argument("--initial_distribution_variance", type=float, default=0.01,
                        help="Variance of the initial latent state distribution z_0 ~ N(0, v)")

    args = parser.parse_args()
    main(x_observed_ID=args.x_observed_ID,
        num_iterations=args.num_iterations,
        N=args.N,
        step_sizes=args.step_sizes,
        theta_0=args.theta_0, 
        sample_trajectories=args.sample_trajectories,
        print_ks=args.print_ks,
        sigma2_alpha=args.sigma2_alpha,
        sigma2_beta=args.sigma2_beta,
        beta2_alpha=args.beta2_alpha,
        beta2_beta=args.beta2_beta,
        rho_lower=args.rho_lower,
        rho_upper=args.rho_upper,
        T=args.T,
        initial_distribution_variance=args.initial_distribution_variance
    )


