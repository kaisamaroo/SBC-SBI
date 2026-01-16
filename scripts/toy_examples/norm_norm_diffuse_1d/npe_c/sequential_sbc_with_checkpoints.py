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


def main(sigma, N_iter, N_samp, num_sequential_rounds, num_simulations_per_round, checkpoint_percent):
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_sbc_ranks{i}.npy"):
        i += 1

    prior = make_prior(sigma)
    N_iter_per_checkpoint = int(N_iter * checkpoint_percent / 100)
    sbc_checkpoint_times = []
    ranks_sequential = []
    samples_dict = {}
    print("\n Running SBC:")
    for checkpoint_id in range(int(100 / checkpoint_percent)):
        print(f"\n Running SBC checkpoint {checkpoint_id+1}:")
        start_time = time.perf_counter()
        ranks_sequential_checkpoint, samples_dict_checkpoint = sbc_ranks_snpe_c_and_samples(simulator,
                                prior,
                                train_snpe_c_posterior,
                                test_function=None,
                                N_iter=N_iter_per_checkpoint,
                                N_samp=N_samp,
                                num_sequential_rounds=num_sequential_rounds,
                                num_simulations_per_round=num_simulations_per_round,
                                show_progress=True)
        end_time = time.perf_counter()
        sbc_time = end_time - start_time
        sbc_checkpoint_times.append(sbc_time)
        ranks_sequential += list(ranks_sequential_checkpoint)
        print(f"\n SBC checkpoint {checkpoint_id+1} completed successfully.")
        for key, value in samples_dict_checkpoint.items():
            samples_dict[key + f"_checkpoint{checkpoint_id}"] = value

        checkpoint_config = {"sigma": sigma, 
            "N_iter": N_iter_per_checkpoint,
            "N_samp": N_samp,
            "num_sequential_rounds": num_sequential_rounds,
            "num_simulations_per_round": num_simulations_per_round,
            "checkpoint_sbc_time": sbc_time}

        # Save checkpoint
        checkpoint_sequential_sbc_save_path = results_path + f"/sequential_sbc_ranks{i}_checkpoint{checkpoint_id}.npy"
        checkpoint_config_save_path = results_path + f"/sequential_sbc_ranks{i}_checkpoint{checkpoint_id}.yaml"
        checkpoint_simulations_save_path = results_path + f"/sequential_sbc_ranks{i}_checkpoint{checkpoint_id}_simulations.npz"
        
        print(f"\n Saving checkpoint ranks to {checkpoint_sequential_sbc_save_path}:")
        np.save(checkpoint_sequential_sbc_save_path, ranks_sequential_checkpoint)
        print("\n Checkpoint ranks saved.")

        print(f"\n Saving checkpoint config file to {checkpoint_config_save_path}:")
        with open(checkpoint_config_save_path, "w") as f:
            yaml.safe_dump(checkpoint_config, f)
        print("\n Checkpoint config file saved successfully.")

        # Save simulations:
        print(f"\n Saving checkpoint simulations to {checkpoint_simulations_save_path}:")
        np.savez(checkpoint_simulations_save_path, **samples_dict_checkpoint)
        print("\n Checkpoint simulations saved successfully.")


    print("\n SBC finished.")

    total_sbc_time = sum(sbc_checkpoint_times)
    ranks_sequential = np.array(ranks_sequential)

    config = {"sigma": sigma, 
              "N_iter": N_iter,
              "N_samp": N_samp,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "total_sbc_time": total_sbc_time}

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
    parser.add_argument("--checkpoint_percent", type=int, default=10)
    args = parser.parse_args()
    main(args.sigma, args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round, args.checkpoint_percent)
