import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.stoch_vol import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time
import math

# flake8: noqa

print(Path(__file__).resolve().parents[5])

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "stoch_vol" / "npe_c")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "stoch_vol" / "data")


def main(x_observed_ID, num_sequential_rounds, num_simulations_per_round,
         use_combined_loss, density_estimator,
         sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower,
         rho_upper, T, initial_distribution_variance):
    
    # Convert use_combined_loss into bool
    use_combined_loss = str(use_combined_loss).lower() not in ["false", "0", "no"]

    # Import data to condition on
    trajectory = np.load(trajectories_path + f"/trajectory{x_observed_ID}.npz")
    x_observed = trajectory["x"]
    x_observed = x_observed[:T+1] # Only take first T+1 samples from trajectory
    x_observed = torch.tensor(x_observed)

    # Dictionary to save each round's posterior
    posteriors_dict = {}

    # SBI training:
    prior = make_prior(sigma2_alpha, sigma2_beta, beta2_alpha, beta2_beta, rho_lower,
         rho_upper, T, initial_distribution_variance)
    inference = NPE_C(prior=prior, density_estimator=density_estimator)
    proposal = prior

    # Initialize simulation and train time lists (one per round)
    simulation_times = []
    training_times = []

    for r in range(num_sequential_rounds):
        print(f"\n Round {r+1}:")
        print("\n Generating samples:")
        sample_start_time = time.perf_counter()
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)
        sample_end_time = time.perf_counter()
        sample_time = sample_end_time - sample_start_time
        simulation_times.append(sample_time)
        print("\n Samples generated")

        print("\n Training proposal:")
        training_start_time = time.perf_counter()
        _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
        sequential_posterior = inference.build_posterior().set_default_x(x_observed)
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
              "use_combined_loss": use_combined_loss,
              "density_estimator": density_estimator,
              "sigma2_alpha": sigma2_alpha,
              "sigma2_beta": sigma2_beta,
              "beta2_alpha": beta2_alpha,
              "beta2_beta": beta2_beta,
              "rho_lower": rho_lower,
              "rho_upper": rho_upper,
              "T": T,
              "initial_distribution_variance": initial_distribution_variance}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_posterior{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + f"/sequential_posterior{i}.yaml"
    posteriors_dict_save_path = results_path + f"/sequential_posterior{i}_posteriors_dict.pkl"

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
    parser.add_argument("--x_observed_ID", type=int, required=True)

    parser.add_argument("--num_sequential_rounds", type=int, default=4,
                        help="Number of sequential rounds")
    parser.add_argument("--num_simulations_per_round", type=int, default=5000,
                        help="Number of simulations per sequential round")
    parser.add_argument("--use_combined_loss", type=str, default="False",
                        help="Whether to use combined loss in SNPE-C")
    parser.add_argument("--density_estimator", type=str, default="maf",
                        help="Type of density estimator to use in SBI")

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
    main(
        x_observed_ID=args.x_observed_ID,
        num_sequential_rounds=args.num_sequential_rounds,
        num_simulations_per_round=args.num_simulations_per_round,
        use_combined_loss=args.use_combined_loss,
        density_estimator=args.density_estimator,
        sigma2_alpha=args.sigma2_alpha,
        sigma2_beta=args.sigma2_beta,
        beta2_alpha=args.beta2_alpha,
        beta2_beta=args.beta2_beta,
        rho_lower=args.rho_lower,
        rho_upper=args.rho_upper,
        T=args.T,
        initial_distribution_variance=args.initial_distribution_variance
    )