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


def train_sequential_posterior(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss):
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
    
    return sequential_posterior, density_estimator, config


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss, num_sequential_rounds_list, num_simulations_per_round_list, d_list):
    multiple_sequential_posteriors_dict = {"sequential_posteriors": [],
                      "density_estimators": [],
                      "configs": []}
    
    both_nr_nspr_are_lists = isinstance(num_sequential_rounds_list, list) and isinstance(num_simulations_per_round_list, list)
    d_is_list = isinstance(d_list, list)

    if both_nr_nspr_are_lists and d_is_list:
        raise ValueError("INVALID INPUT TYPES: EITHER BOTH num_sequential_rounds_list AND num_simulations_per_round_list MUST BE LISTS, XOR d_list MUST BE A LIST! Currently, both are true!")
    elif both_nr_nspr_are_lists:
        for num_sequential_rounds, num_simulations_per_round in zip(num_sequential_rounds_list, num_simulations_per_round_list):
            sequential_posterior, density_estimator, config = train_sequential_posterior(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss)
            multiple_sequential_posteriors_dict["sequential_posteriors"].append(sequential_posterior)
            multiple_sequential_posteriors_dict["density_estimators"].append(density_estimator)
            multiple_sequential_posteriors_dict["configs"].append(config)
    elif d_is_list:
        for d in d_list:
            sequential_posterior, density_estimator, config = train_sequential_posterior(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss)
            multiple_sequential_posteriors_dict["sequential_posteriors"].append(sequential_posterior)
            multiple_sequential_posteriors_dict["density_estimators"].append(density_estimator)
            multiple_sequential_posteriors_dict["configs"].append(config)
    else:
        raise ValueError("INVALID INPUT TYPES: EITHER BOTH num_sequential_rounds_list AND num_simulations_per_round_list MUST BE LISTS, XOR d_list MUST BE A LIST! Currently, neither are true!")
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/multiple_sequential_posteriors_dict{i}.pkl"):
        i += 1

    # Define save paths
    multiple_sequential_posteriors_dict_save_path = results_path + f"/multiple_sequential_posteriors_dict{i}.pkl"

    print(f"\n Saving dictionary of posteriors, density estimators, and configs to {multiple_sequential_posteriors_dict_save_path}:")
    with open(multiple_sequential_posteriors_dict_save_path, "wb") as handle:
        pickle.dump(multiple_sequential_posteriors_dict, handle)
    print("\n Saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--x_observed", type=float, required=True)
    parser.add_argument("--num_sequential_rounds_list", type=int, nargs="+", default=None)
    parser.add_argument("--num_simulations_per_round_list", type=int, nargs="+", default=None)
    parser.add_argument("--d_list", type=int, nargs="+", default=None)

    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--use_combined_loss", type=bool, default=False)

    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds, args.num_simulations_per_round, args.d, args.L, args.U, args.use_combined_loss, args.num_sequential_rounds_list, args.num_simulations_per_round_list, args.d_list)