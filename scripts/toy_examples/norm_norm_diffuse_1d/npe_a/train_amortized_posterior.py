import torch
from sbi.inference import NPE_A
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
import argparse
from pathlib import Path
import os
import time
import yaml

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_a")


def main(sigma, num_simulations, num_components):
    print("Training posterior:")
    prior = make_prior(sigma)
    inference = NPE_A(prior=prior, num_components=num_components)  # NPE-A algorithm (fast epsilon-free inference)
    print("\n Generating samples:")
    t0 = time.perf_counter()
    parameter_samples = prior.sample((num_simulations,))  # simulate parameters from prior
    data_samples = simulator(parameter_samples)  # simulate data for each parameter
    print("\n Samples generated successfully.")
    t1 = time.perf_counter()
    print("\n Training posterior:")
    inference = inference.append_simulations(parameter_samples, data_samples)
    density_estimator = inference.train(final_round=True) # final_round=True to ensure MOG density
    amortized_posterior = inference.build_posterior()
    t2 = time.perf_counter()
    print("\n Posterior trained successfully.")

    # Compute runtimes
    simulation_time = t1 - t0
    training_time = t2 - t1
    total_time = simulation_time + training_time

    config = {"sigma": sigma,
            "num_simulations": num_simulations,
            "num_components": num_components,
            "simulation_time": simulation_time,
            "training_time": training_time,
            "total_time": total_time}

    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_posterior{i}.pkl"):
        i += 1

    # Define save paths
    amortized_posterior_save_path = results_path + f"/amortized_posterior{i}.pkl"
    config_save_path = results_path + f"/amortized_posterior{i}.yaml"
    simulations_save_path = results_path + f"/amortized_posterior{i}_simulations.npz"

    print(f"\n Saving trained posterior to {amortized_posterior_save_path}:")
    with open(amortized_posterior_save_path, "wb") as handle:
        pickle.dump(amortized_posterior, handle)
    print("\n Posterior saved successfully.")

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    # Save simulations:
    print(f"\n Saving simulations to {simulations_save_path}:")
    np.savez(simulations_save_path,
             parameter_samples=parameter_samples,
             data_samples=data_samples)
    print("\n Simulations saved successfully.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_simulations", type=int, default=20000)
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--num_components", type=int, default=1)
    args = parser.parse_args()
    main(args.sigma, args.num_simulations, args.num_components)