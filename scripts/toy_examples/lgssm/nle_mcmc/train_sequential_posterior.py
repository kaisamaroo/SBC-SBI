import torch
from sbi.inference import NLE
import numpy as np
import pickle
from examples.lgssm import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time
import math

# flake8: noqa

print(Path(__file__).resolve().parents[5])

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "nle_mcmc")
trajectories_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "data")


def main(x_observed_ID, num_sequential_rounds, num_simulations_per_round,
         mcmc_method, density_estimator,
         tau_loc, tau_scale, tau_lower, tau_upper, rho_lower,
         rho_upper, T):
    
    # Import data to condition on
    trajectory = np.load(trajectories_path + f"/trajectory{x_observed_ID}.npz")
    x_observed = trajectory["x"]
    x_observed = x_observed[:T+1] # Only take first T+1 samples from trajectory
    x_observed = torch.tensor(x_observed)

    with open(trajectories_path + f"/trajectory{x_observed_ID}.yaml", "r") as f:
        trajectory_config = yaml.safe_load(f)
    sigma_true = trajectory_config["sigma_true"]

    # Dictionary to save each round's posterior
    posteriors_dict = {}

    # SBI training:
    prior = make_prior(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T)
    inference = NLE(prior=prior, density_estimator=density_estimator)
    proposal = prior

    # Initialize simulation and train time lists (one per round)
    simulation_times = []
    training_times = []

    for r in range(num_sequential_rounds):
        print(f"\n Round {r+1}:")
        print("\n Generating samples:")
        sample_start_time = time.perf_counter()
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples, sigma_true)
        sample_end_time = time.perf_counter()
        sample_time = sample_end_time - sample_start_time
        simulation_times.append(sample_time)
        print("\n Samples generated")

        print("\n Training proposal:")
        training_start_time = time.perf_counter()
        _ = inference.append_simulations(parameter_samples, data_samples).train()
        sequential_posterior = inference.build_posterior(sample_with="mcmc", mcmc_method=mcmc_method).set_default_x(x_observed)
        posteriors_dict[f"round_{r}"] = sequential_posterior
        proposal = sequential_posterior
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        training_times.append(training_time)
        print("\n Proposal trained successfully:")
    print("\n Posterior trained successfully.")

    config = {"x_observed_ID": x_observed_ID,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "simulation_times": simulation_times,
              "training_times": training_times,
              "mcmc_method": mcmc_method,
              "density_estimator": density_estimator,
              "rho_lower": rho_lower,
              "rho_upper": rho_upper,
              "T": T,
              "tau_loc": tau_loc,
              "tau_scale": tau_scale,
              "tau_lower": tau_lower,
              "tau_upper": tau_upper,
              }
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_posterior_T{T}_xobsid{x_observed_ID}_{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + f"/sequential_posterior_T{T}_xobsid{x_observed_ID}_{i}.yaml"
    posteriors_dict_save_path = results_path + f"/sequential_posterior_T{T}_xobsid{x_observed_ID}_{i}_posteriors_dict.pkl"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving posteriors to {posteriors_dict_save_path}:")
    with open(posteriors_dict_save_path, "wb") as f:
        pickle.dump(posteriors_dict, f)
    print(f"\n Posteriors saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_observed_ID", type=int, default=0)

    parser.add_argument("--num_sequential_rounds", type=int, default=8,
                        help="Number of sequential rounds")
    parser.add_argument("--num_simulations_per_round", type=int, default=5000,
                        help="Number of simulations per sequential round")
    parser.add_argument("--mcmc_method", type=str, default="slice_np_vectorized")
    parser.add_argument("--density_estimator", type=str, default="maf",
                        help="Type of density estimator to use in SBI")
    
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
    main(
        x_observed_ID=args.x_observed_ID,
        num_sequential_rounds=args.num_sequential_rounds,
        num_simulations_per_round=args.num_simulations_per_round,
        mcmc_method=args.mcmc_method,
        density_estimator=args.density_estimator,
        tau_loc=args.tau_loc,
        tau_scale=args.tau_scale,
        tau_lower=args.tau_lower,
        tau_upper=args.tau_upper,
        rho_lower=args.rho_lower,
        rho_upper=args.rho_upper,
        T=args.T,
    )