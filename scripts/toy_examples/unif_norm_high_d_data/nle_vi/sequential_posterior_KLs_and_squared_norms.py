import torch
from sbi.inference import NLE
import numpy as np
import pickle
from examples.unif_norm_high_d_data import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm_high_d_data" / "nle_vi")
hadamard_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm_high_d_data" / "hadamard_matrices")


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round,
         d, k, L, U, density_estimator, num_repetitions, vi_method):
    
    # Only works if we are up-projecting
    if k < d:
            raise AssertionError(f"k MUST BE LARGER THAN d! Currently k = {k} and d = {d}.")
    
    hadamard_matrix_path = hadamard_path + f"/square_{k}.npy" 
    hadamard_matrix = np.load(hadamard_matrix_path) # (k, k) Hadamard matrix
    hadamard_matrix = torch.tensor(hadamard_matrix)
    
    # Default x_observed is the d+k vector of zeros (i.e. of x_observed not passed in command line)
    if x_observed == []:
        x_observed = [0 for _ in range(k)]

    # Store each squared_norms as a row of a np array with num_repetitions rows
    squared_norms_dict = {f"round_{r}": np.zeros((num_repetitions, num_simulations_per_round)) for r in range(num_sequential_rounds)}
    
    # Initialize simulation and train time lists (averaged over num_repetitions)
    avg_simulation_times = {f"round_{r}": 0 for r in range(num_sequential_rounds)}
    avg_training_times = {f"round_{r}": 0 for r in range(num_sequential_rounds)}

    for rep in range(num_repetitions):
        print(f"\n Repetition {rep}:")
        prior = make_prior(L=L, U=U, d=d)
        inference = NLE(prior=prior, density_estimator=density_estimator)
        proposal = prior

        for r in range(num_sequential_rounds):
            print(f"\n Round {r+1}:")
            print("\n Generating samples:")
            sample_start_time = time.perf_counter()
            parameter_samples = proposal.sample((num_simulations_per_round,))
            data_samples = simulator(parameter_samples, hadamard_matrix, sigma=sigma, d=d)
            sample_end_time = time.perf_counter()
            sample_time = sample_end_time - sample_start_time
            avg_simulation_times[f"round_{r}"] += sample_time / num_repetitions
            print("\n Samples generated")

            # Don't compute squared norms and KLs for the prior
            if r > 0:
                print("\n Computing squared norms:")
                # Save squared norms of parameter samples 
                squared_norms = (parameter_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
                squared_norms_dict[f"round_{r-1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
                print("\n Squared norms generated")

            print("\n Training proposal:")
            training_start_time = time.perf_counter()
            ########## TRAINING (no need for round split)
            density_estimator_ = inference.append_simulations(parameter_samples, data_samples).train()
            sequential_posterior = inference.build_posterior(sample_with="vi", vi_method=vi_method).set_default_x(torch.tensor(x_observed)).train()
            proposal = sequential_posterior
            ##########
            training_end_time = time.perf_counter()
            training_time = training_end_time - training_start_time
            avg_training_times[f"round_{r}"] += training_time / num_repetitions
            print("\n Proposal trained successfully:")
        print("\n Posterior trained successfully.")

        # Append final posterior squared norm and KL
        parameter_samples = sequential_posterior.sample((num_simulations_per_round,))

        print("\n Computing final round squared norms:")
        # Save squared norms of parameter samples 
        squared_norms = (parameter_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
        squared_norms_dict[f"round_{num_sequential_rounds-1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
        print("\n Squared norms generated")

    config = {"sigma": sigma, 
              "x_observed": x_observed,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "avg_simulation_times": avg_simulation_times,
              "avg_training_times": avg_training_times,
              "d": d,
              "k": k,
              "L": L,
              "U": U,
              "density_estimator": density_estimator,
              "num_repetitions": num_repetitions,
              "vi_method": vi_method}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/KLs_squared_norms{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + f"/KLs_squared_norms{i}.yaml"
    squared_norms_save_path = results_path + f"/KLs_squared_norms{i}_squared_norms_dict.npz"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving squared norms to {squared_norms_save_path}:")
    np.savez(squared_norms_save_path, **squared_norms_dict)
    print("\n Squared norms saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--x_observed", type=float, nargs="+", default=[])
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--density_estimator", type=str, default="maf")
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--vi_method", type=str, default="rKL")
    
    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds,
         args.num_simulations_per_round, args.d, args.k, args.L, args.U,
         args.density_estimator, args.num_repetitions, args.vi_method)