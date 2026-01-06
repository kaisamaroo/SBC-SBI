import torch
from sbi.inference import NPE_A
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
import argparse
from pathlib import Path

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_a")

def main(sigma, num_simulations, num_components):
    print("Training posterior:")
    prior = make_prior(sigma)
    inference = NPE_A(prior=prior, num_components=num_components)  # NPE-A algorithm (fast epsilon-free inference)
    mu = prior.sample((num_simulations,))  # simulate parameters from prior
    x = simulator(mu)  # simulate data for each parameter
    inference = inference.append_simulations(mu, x)
    density_estimator = inference.train(final_round=True) # final_round=True to ensure MOG density
    amortized_posterior = inference.build_posterior()
    print("Posterior trained successfully")

    # Save posterior using pickle
    save_path = results_path + f"/amortized_posterior_sig{sigma}_ns{num_simulations}_nc{num_components}.pkl"
    print(f"Saving trained posterior to {save_path}:")
    with open(save_path, "wb") as handle:
        pickle.dump(amortized_posterior, handle)
    print("Posterior saved successfully.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_simulations", type=int, default=20000)
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--num_components", type=int, default=1)
    args = parser.parse_args()
    main(args.sigma, args.num_simulations, args.num_components)