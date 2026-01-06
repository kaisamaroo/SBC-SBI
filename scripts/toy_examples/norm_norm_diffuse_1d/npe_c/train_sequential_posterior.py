import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
import argparse
from pathlib import Path

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_c")


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, show_progress):
    prior = make_prior(sigma)
    inference = NPE_C(prior)
    proposal = prior
    for r in range(num_sequential_rounds):
        if show_progress:
            print(f"Round {r+1}:")
            print("Generating samples:")
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)
        if show_progress:  
            print("Samples generated")
        if r == num_sequential_rounds - 1:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train()
            sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train()
            sequential_posterior = inference.build_posterior().set_default_x(x_observed)
            proposal = sequential_posterior
    print("Posterior trained successfully.")
    
    # Save posterior using pickle
    save_path = results_path + f"/sequential_posterior_xobs{x_observed}_sig{sigma}_nspr{num_simulations_per_round}_nr{num_sequential_rounds}.pkl"
    print(f"Saving trained posterior to {save_path}:")
    with open(save_path, "wb") as handle:
        pickle.dump(sequential_posterior, handle)
    print("Posterior saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--x_observed", type=float, required=True)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--show_progress", type=bool, default=True)
    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds, args.num_simulations_per_round, args.show_progress)