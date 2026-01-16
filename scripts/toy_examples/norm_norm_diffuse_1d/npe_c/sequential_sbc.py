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


def main(sigma, N_iter, N_samp, num_sequential_rounds, num_simulations_per_round):
    prior = make_prior(sigma)
    print("\n Running SBC:")
    start_time = time.perf_counter()
    ranks_sequential, samples_dict = sbc_ranks_snpe_c_and_samples(simulator,
                            prior,
                            train_snpe_c_posterior,
                            test_function=None,
                            N_iter=N_iter,
                            N_samp=N_samp,
                            num_sequential_rounds=num_sequential_rounds,
                            num_simulations_per_round=num_simulations_per_round,
                            show_progress=True)
    end_time = time.perf_counter()
    total_sbc_time = end_time - start_time
    ranks_sequential = np.array(ranks_sequential)
    print("\n SBC finished.")

    config = {"sigma": sigma, 
              "N_iter": N_iter,
              "N_samp": N_samp,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "total_sbc_time": total_sbc_time}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_sbc_ranks{i}.npy"):
        i += 1

    sequential_sbc_save_path = results_path + f"/sequential_sbc_ranks{i}.npy"
    config_save_path = results_path + f"/sequential_sbc_ranks{i}.yaml"
    simulations_save_path = results_path + f"/sequential_sbc_ranks{i}_simulations.npz"
    
    print(f"\n Saving ranks to {sequential_sbc_save_path}:")
    np.save(sequential_sbc_save_path, ranks_sequential)
    print("\n Ranks saved.")

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
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    args = parser.parse_args()
    main(args.sigma, args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round)
