import torch
from sbi.inference import NRE
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
results_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "nre")
trajectories_path = str(path_to_repo / "results" / "toy_examples" / "lgssm" / "data")

# Density_estimator not needed and could be removed!
def main(x_observed_ID, num_sequential_rounds, num_simulations_per_round,
         classifier, mcmc_method, density_estimator,
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

    # Dictionary to save each round's posterior samples
    rho_samples_dict = {}
    tau_samples_dict = {}
    latent_states_samples_dict = {}

    # SBI training:
    prior = make_prior(rho_lower, rho_upper, tau_loc, tau_scale, tau_lower, tau_upper, T)
    inference = NRE(prior=prior, classifier=classifier)
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

        if r > 0: # Don't save prior samples
            rho_samples_dict[f"round_{r}"] = parameter_samples[:2000, 0].cpu().numpy()
            tau_samples_dict[f"round_{r}"] = parameter_samples[:2000, 1].cpu().numpy()
            latent_states_samples_dict[f"round_{r}"] = parameter_samples[:2000, 2:].cpu().numpy()

        print("\n Training proposal:")
        training_start_time = time.perf_counter()
        _ = inference.append_simulations(parameter_samples, data_samples).train()
        sequential_posterior = inference.build_posterior(sample_with="mcmc", mcmc_method=mcmc_method).set_default_x(x_observed)
        proposal = sequential_posterior
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        training_times.append(training_time)
        print("\n Proposal trained successfully:")
    print("\n Posterior trained successfully.")

    # Sample from final round posterior
    print("\n Generating samples from final posterior:")
    parameter_samples = sequential_posterior.sample((num_simulations_per_round,))
    print("\n Samples generated")
    rho_samples_dict[f"round_{num_sequential_rounds}"] = parameter_samples[:2000, 0].cpu().numpy()
    tau_samples_dict[f"round_{num_sequential_rounds}"] = parameter_samples[:2000, 1].cpu().numpy()
    latent_states_samples_dict[f"round_{num_sequential_rounds}"] = parameter_samples[:2000, 2:].cpu().numpy()

    config = {"x_observed_ID": x_observed_ID,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "simulation_times": simulation_times,
              "training_times": training_times,
              "density_estimator": density_estimator,
              "classifier": classifier,
              "mcmc_method": mcmc_method,
              "rho_lower": rho_lower,
              "rho_upper": rho_upper,
              "T": T,
              "tau_loc": tau_loc,
              "tau_scale": tau_scale,
              "tau_lower": tau_lower,
              "tau_upper": tau_upper,
              }
    
    if num_sequential_rounds == 1:
        method = "amortized"
    else:
        method = "sequential"
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + "/" + method + f"_posterior_T{T}_xobsid{x_observed_ID}__{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + "/" + method + f"_posterior_T{T}_xobsid{x_observed_ID}__{i}.yaml"
    rho_samples_save_path = results_path + "/" + method + f"_posterior_T{T}_xobsid{x_observed_ID}__{i}_rho_samples.npz"
    tau_samples_save_path = results_path + "/" + method + f"_posterior_T{T}_xobsid{x_observed_ID}__{i}_tau_samples.npz"
    latent_states_samples_save_path = results_path + "/" + method + f"_posterior_T{T}_xobsid{x_observed_ID}__{i}_latent_states_samples.npz"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving rho samples to {rho_samples_save_path}:")
    np.savez(rho_samples_save_path, **rho_samples_dict)
    print(f"\n Rho samples saved successfully.")

    print(f"\n Saving tau samples to {tau_samples_save_path}:")
    np.savez(tau_samples_save_path, **tau_samples_dict)
    print(f"\n Tau samples saved successfully.")

    print(f"\n Saving latent states samples to {latent_states_samples_save_path}:")
    np.savez(latent_states_samples_save_path, **latent_states_samples_dict)
    print(f"\n Latent states samples saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_observed_ID", type=int, default=0)

    parser.add_argument("--num_sequential_rounds", type=int, default=8,
                        help="Number of sequential rounds")
    parser.add_argument("--num_simulations_per_round", type=int, default=5000,
                        help="Number of simulations per sequential round")
    parser.add_argument("--density_estimator", type=str, default="maf",
                        help="Type of density estimator to use in SBI")
    parser.add_argument("--classifier", type=str, default="resnet")
    parser.add_argument("--mcmc_method", type=str, default="slice_np_vectorized")
    
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
        classifier=args.classifier,
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