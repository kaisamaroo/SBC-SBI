import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
from sbc.sbc_tools import sbc_ranks_snpe_c_and_samples, train_snpe_c_posterior
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_c")


def main(sigma, N_iter, N_samp, num_sequential_rounds, num_simulations_per_round, experiment_ID):
    # By default, experiment_ID is -1, meaning we start a new experiment ID.
    continue_experiment = experiment_ID >= 0
    if continue_experiment:
        # Continue with existing experiment
        i = experiment_ID
        if not os.path.exists(results_path + f"/sequential_sbc_ranks{i}.npy"):
            raise AssertionError("No file in directory " + results_path + f"/sequential_sbc_ranks{i}.npy")
        print(f"\n Continuing to append to experiment {experiment_ID}.")
        sequential_sbc_path = results_path + f"/sequential_sbc_ranks{i}.npy"
        config_path = results_path + f"/sequential_sbc_ranks{i}.yaml"
        simulations_path = results_path + f"/sequential_sbc_ranks{i}_simulations.npz"

    prior = make_prior(sigma)
    print("\n Running SBC:")
    for n in range(N_iter):
        print(f"\n SBC round {n} out of {N_iter}")
        print("\n Generating rank")
        start_time = time.perf_counter()
        rank_sequential, sample_dict = sbc_ranks_snpe_c_and_samples(simulator,
                                prior,
                                train_snpe_c_posterior,
                                test_function=None,
                                N_iter=1,
                                N_samp=N_samp,
                                num_sequential_rounds=num_sequential_rounds,
                                num_simulations_per_round=num_simulations_per_round,
                                show_progress=False)
        end_time = time.perf_counter()
        print("\n Rank generated")
        sbc_time = end_time - start_time
        rank_sequential = rank_sequential[0] # Get float rank
        if (n == 0) and (not continue_experiment):
            config = {"sigma": sigma, 
                "N_iter": 1,
                "N_samp": N_samp,
                "num_sequential_rounds": num_sequential_rounds,
                "num_simulations_per_round": num_simulations_per_round,
                "sbc_times": [sbc_time],
                "total_sbc_time": sbc_time}
            
            # Find next available ID
            i = 0
            while os.path.exists(results_path + f"/sequential_sbc_ranks{i}.npy"):
                i += 1
            sequential_sbc_path = results_path + f"/sequential_sbc_ranks{i}.npy"
            config_path = results_path + f"/sequential_sbc_ranks{i}.yaml"
            simulations_path = results_path + f"/sequential_sbc_ranks{i}_simulations.npz"

            print(f"\n Saving first config file to {config_path}:")
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)
            print("\n Config file saved successfully.")
            
            print(f"\n Saving first rank to {sequential_sbc_path}:")
            np.save(sequential_sbc_path, np.array([rank_sequential]).reshape(-1))
            print("\n Rank saved.")

            # Save simulations:
            print(f"\n Saving first simulations to {simulations_path}:")
            np.savez(simulations_path, **sample_dict)
            print("\n Simulations saved successfully.")

        else:
            # Append to existing files

            # Load config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = dict(config)
            # Assert that hyperparameters are equal
            # If we are appending to an existing experiment, we must assert that
            # all hyperparameters are equal. If not, it's not sensible to append simulations.
            assert(
                config["N_samp"] == N_samp
                and config["sigma"] == sigma
                and config["num_sequential_rounds"] == num_sequential_rounds
                and config["num_simulations_per_round"] == num_simulations_per_round
            )
            old_N_iter = config["N_iter"] # For simulation indexing
            # Update config
            config["N_iter"] += 1
            config["sbc_times"].append(sbc_time)
            config["total_sbc_time"] += sbc_time
            # Save updated config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

            # Load ranks
            sequential_ranks = np.load(sequential_sbc_path)
            # Update ranks by appending new rank
            sequential_ranks = list(sequential_ranks)
            sequential_ranks.append(rank_sequential)
            # Save updated ranks
            np.save(sequential_sbc_path, np.array(sequential_ranks))

            # Load simulations
            samples_dict = np.load(simulations_path)
            samples_dict = dict(samples_dict)
            # Append new simulations
            samples_dict[f"prior_sample_round_{old_N_iter}"] = sample_dict["prior_sample_round_0"]
            samples_dict[f"data_sample_round_{old_N_iter}"] = sample_dict["data_sample_round_0"]
            samples_dict[f"posterior_samples_round_{old_N_iter}"] = sample_dict["posterior_samples_round_0"]
            # Save simulations
            np.savez(simulations_path, **samples_dict)
    print("\n SBC finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--experiment_ID", type=int, default=-1) # By default, start new experiment
    args = parser.parse_args()
    main(args.sigma, args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round, args.experiment_ID)
