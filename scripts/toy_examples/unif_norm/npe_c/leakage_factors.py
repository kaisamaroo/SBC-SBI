import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
import argparse
from pathlib import Path
import os
import time
import yaml

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "npe_c")


def train_amortized_posterior(sigma, num_simulations, d, L, U, force_first_round_loss):
    prior = make_prior(L=L, U=U, d=d)
    inference = NPE_C(prior=prior)
    print("\n Generating samples:")
    t0 = time.perf_counter()
    parameter_samples = prior.sample((num_simulations,))  # shape (num_simulations, d)
    data_samples = simulator(parameter_samples, sigma=sigma, d=d)  # shape (num_simulations, nd)
    print("\n Samples generated successfully.")
    t1 = time.perf_counter()
    print("\n Training posterior:")
    inference = inference.append_simulations(parameter_samples, data_samples)
    density_estimator = inference.train(force_first_round_loss=force_first_round_loss)
    amortized_posterior = inference.build_posterior()
    t2 = time.perf_counter()
    print("\n Posterior trained successfully.")

    # Compute runtimes
    simulation_time = t1 - t0
    training_time = t2 - t1
    total_time = simulation_time + training_time

    config = {"sigma": sigma,
            "num_simulations": num_simulations,
            "simulation_time": simulation_time,
            "training_time": training_time,
            "total_time": total_time,
            "d": d,
            "L": L,
            "U": U,
            "force_first_round_loss": force_first_round_loss}
    
    return amortized_posterior, density_estimator, config


def main(sigma, num_simulations_list, n, d, L, U):
    multiple_amortized_posteriors_dict = {"amortized_posteriors": [],
                      "density_estimators": [],
                      "configs": []}

    for num_simulations in num_simulations_list:
        amortized_posterior, density_estimator, config = train_amortized_posterior(sigma, num_simulations, n, d, L, U)
        multiple_amortized_posteriors_dict["amortized_posteriors"].append(amortized_posterior)
        multiple_amortized_posteriors_dict["density_estimators"].append(density_estimator)
        multiple_amortized_posteriors_dict["configs"].append(config)
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/multiple_amortized_posteriors_dict{i}.pkl"):
        i += 1

    # Define save paths
    multiple_amortized_posteriors_dict_save_path = results_path + f"/multiple_amortized_posteriors_dict{i}.pkl"

    print(f"\n Saving dictionary of posteriors, density estimators, and configs to {multiple_amortized_posteriors_dict_save_path}:")
    with open(multiple_amortized_posteriors_dict_save_path, "wb") as handle:
        pickle.dump(multiple_amortized_posteriors_dict, handle)
    print("\n Saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_simulations_list", type=list, default=[10, 25, 50, 75, 100, 200, 500, 1000])
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--force_first_round_loss", type=bool, default=False)
    
    args = parser.parse_args()
    main(args.sigma, args.num_simulations_list, args.d, args.L, args.U, args.force_first_round_loss)