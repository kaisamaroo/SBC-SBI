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
from examples.lgssm import metropolis_hastings
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
results_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "kalman_filter")
trajectories_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "data")


def main(x_observed_ID, num_iterations, step_sizes, rho_0, tau_0, rho_lower,
         rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T):

    # Import data to condition on
    trajectory = np.load(trajectories_path + f"/trajectory{x_observed_ID}.npz")
    x_observed = trajectory["x"]
    x_observed = x_observed[:T+1] # Only take first T+1 samples from trajectory

    with open(trajectories_path + f"/trajectory{x_observed_ID}.yaml", "r") as f:
        trajectory_config = yaml.safe_load(f)
    sigma_true = trajectory_config["sigma_true"]
    rho_true = trajectory_config["rho_true"]
    tau_true = trajectory_config["tau_true"]

    if rho_0 is None:
        rho_0 = rho_true
    if tau_0 is None:
        tau_0 = tau_true

    print("\n STARTING SAMPLING:")
    start_time = time.perf_counter()

    rho_samples, tau_samples, z_samples, rejection_rate = metropolis_hastings(rho_0, tau_0, x_observed, 
                                                                               sigma_true, step_sizes, num_iterations,
                                                                               rho_lower, rho_upper, tau_loc,
                                                                               tau_scale, tau_lower, tau_upper)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("\n PMMH FINISHED.")

    posterior_samples_dict = {}
    posterior_samples_dict["rho_samples"] = rho_samples
    posterior_samples_dict["tau_samples"] = tau_samples
    posterior_samples_dict["z_samples"] = z_samples

    config = {"x_observed_ID": x_observed_ID,
              "total_time": total_time,
              "rho_lower": rho_lower,
              "rho_upper": rho_upper,
              "T": T,
              "num_iterations": num_iterations,
              "step_sizes": step_sizes,
              "rho_0": rho_0,
              "tau_0": tau_0,
              "sigma_true": sigma_true,
              "tau_loc": tau_loc,
              "tau_scale": tau_scale,
              "tau_lower": tau_lower,
              "tau_upper": tau_upper,
              "rejection_rate": rejection_rate}
        
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
    parser.add_argument("--x_observed_ID", type=int, default=0)

    parser.add_argument("--num_iterations", type=int, default=20000,
                            help="Number of PMMH iterations (not including initial iterate)")
    parser.add_argument("--step_sizes", nargs=2, type=float, default=[0.1, 0.1], # These are very rough defaults
                            help="Step sizes for sigma^2, beta^2, rho in PMMH")
    parser.add_argument("--rho_0", type=float, default=None,
                            help="Initial parameter iterate for rho. If None, it initializes to the true value.")
    parser.add_argument("--tau_0", type=float, default=None,
                            help="Initial parameter iterate for tau. If None, it initializes to the true value.")

    parser.add_argument("--rho_lower", type=float, default=0.0,
                        help="Lower bound of Uniform prior on rho")
    parser.add_argument("--rho_upper", type=float, default=1.0,
                        help="Upper bound of Uniform prior on rho")
    parser.add_argument("--tau_loc", type=float, default=1.0,
                        help="Mean of truncated gaussian prior over tau")
    parser.add_argument("--tau_scale", type=float, default=1.0,
                        help="Scale of truncated gaussian prior over tau")
    parser.add_argument("--tau_lower", type=float, default=0,
                        help="Lower bound of truncated gaussian prior over tau")
    parser.add_argument("--tau_upper", type=float, default=2.0,
                        help="Upper bound of truncated gaussian prior over tau")

    parser.add_argument("--T", type=int, required=True,
                        help="Length of observed time series (number of steps)")

    args = parser.parse_args()
    main(x_observed_ID = args.x_observed_ID,
         num_iterations = args.num_iterations,
         step_sizes = args.step_sizes,
         rho_0 = args.rho_0,
         tau_0 = args.tau_0,
         rho_lower = args.rho_lower,
         rho_upper = args.rho_upper,
         tau_loc = args.tau_loc,
         tau_scale = args.tau_scale,
         tau_lower = args.tau_lower, 
         tau_upper = args.tau_upper,
         T = args.T)
