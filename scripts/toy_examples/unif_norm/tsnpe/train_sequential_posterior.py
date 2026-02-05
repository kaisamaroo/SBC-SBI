import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "npe_c")


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss):
    prior = make_prior(L=L, U=U, d=d)
    inference = NPE_C(prior=prior)
    proposal = prior

    # Initialize simulation and train time lists (one per round)
    simulation_times = []
    training_times = []

    # Initialize samples_dict to store all parameter and data samples
    samples_dict = {}

    for r in range(num_sequential_rounds):
        print(f"\n Round {r+1}:")
        print("\n Generating samples:")
        sample_start_time = time.perf_counter()
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples, sigma=sigma, d=d)
        sample_end_time = time.perf_counter()
        sample_time = sample_end_time - sample_start_time
        simulation_times.append(sample_time)
        # Save samples
        samples_dict[f"parameter_samples_round_{r}"] = parameter_samples
        samples_dict[f"data_samples_round_{r}"] = data_samples
        print("\n Samples generated")

        print("\n Training proposal:")
        training_start_time = time.perf_counter()
        if r == num_sequential_rounds - 1:
            density_estimator = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            sequential_posterior = inference.build_posterior().set_default_x(x_observed)
            proposal = sequential_posterior
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        training_times.append(training_time)
        print("\n Proposal trained successfully:")
    print("\n Posterior trained successfully.")

    total_time = sum(simulation_times) + sum(training_times)

    config = {"sigma": sigma, 
              "x_observed": x_observed,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "simulation_times": simulation_times,
              "training_times": training_times,
              "total_time": total_time,
              "d": d,
              "L": L,
              "U": U,
              "use_combined_loss": use_combined_loss}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_posterior{i}.pkl"):
        i += 1
    
    # Save paths
    sequential_posterior_save_path = results_path + f"/sequential_posterior{i}.pkl"
    sequential_density_estimator_path = results_path + f"/sequential_posterior{i}_density_estimator.pkl"
    config_save_path = results_path + f"/sequential_posterior{i}.yaml"
    simulations_save_path = results_path + f"/sequential_posterior{i}_simulations.npz"

    print(f"\n Saving trained posterior to {sequential_posterior_save_path}:")
    with open(sequential_posterior_save_path, "wb") as handle:
        pickle.dump(sequential_posterior, handle)
    print("\n Posterior saved successfully.")

    print(f"\n Saving trained density estimator to {sequential_density_estimator_path}:")
    with open(sequential_density_estimator_path, "wb") as handle:
        pickle.dump(density_estimator, handle)
    print("\n Density estimator saved successfully.")

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    # Save simulations:
    print(f"\n Saving simulations to {simulations_save_path}:")
    np.savez(simulations_save_path, **samples_dict)
    print("\n Simulations saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--x_observed", type=float, required=True)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--use_combined_loss", type=bool, default=False)

    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds,
         args.num_simulations_per_round, args.d, args.L, args.U, args.use_combined_loss)