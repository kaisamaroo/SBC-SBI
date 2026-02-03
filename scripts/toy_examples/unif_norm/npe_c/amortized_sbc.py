import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
from sbc.sbc_tools import sbc_ranks_and_samples
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "npe_c")


def main(N_iter, N_samp, sigma, amortized_posterior_ID, d, L, U):
    prior = make_prior(L=L, U=U, d=d)
    amortized_posterior_path = results_path + f"/amortized_posterior{amortized_posterior_ID}.pkl"
    # Retrieve amortized posterior
    with open(amortized_posterior_path, "rb") as f:
        amortized_posterior = pickle.load(f)
    print("\n Running SBC:")
    start_time = time.perf_counter()
    # Get simulator in required form
    simulator_ = lambda x: simulator(x, sigma=sigma, d=d)
    ranks, samples_dict = sbc_ranks_and_samples(model=simulator_, prior=prior, posterior=amortized_posterior, N_iter=N_iter, N_samp=N_samp, show_progress=True)
    end_time = time.perf_counter()
    print("\n SBC finished.")
    ranks = np.array(ranks)
    total_sbc_time = end_time - start_time

    config = {"N_iter": N_iter,
              "N_samp": N_samp,
              "sigma": sigma,
              "amortized_posterior_ID": amortized_posterior_ID,
              "total_sbc_time": total_sbc_time,
              "d": d,
              "L": L,
              "U": U}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.npy"):
        i += 1

    sbc_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.npy"
    config_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.yaml"
    simulations_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}_simulations.npz"

    print(f"\n Saving ranks to {sbc_save_path}:")
    np.save(sbc_save_path, ranks)
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
    parser.add_argument("--amortized_posterior_ID", type=int, required=True)
    parser.add_argument("--N_iter", type=int, default=1000)
    parser.add_argument("--N_samp", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    args = parser.parse_args()
    main(args.N_iter, args.N_samp, args.sigma, args.amortized_posterior_ID,
         args.d, args.L, args.U)
