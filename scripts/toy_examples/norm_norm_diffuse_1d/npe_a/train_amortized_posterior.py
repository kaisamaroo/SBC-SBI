import torch
from torch.distributions import Normal
from sbi.inference import NPE_A
from sbi.analysis import pairplot
import numpy as np
import pickle
import examples
import os
from examples.norm_norm_diffuse_1d import prior_pdf, likelihood_pdf, posterior_pdf, plot_approximate_posterior, make_prior, simulator
import argparse

def main(sigma, num_simulations, save_path):
    prior = make_prior(sigma)
    inference = NPE_A(prior=prior, num_components=1)  # (S)NPE-A algorithm (fast epsilon-free inference)
    mu = prior.sample((num_simulations,))  # simulate parameters from prior
    x = simulator(mu)  # simulate data for each parameter
    inference = inference.append_simulations(mu, x)
    density_estimator = inference.train(final_round=True)
    amortized_posterior = inference.build_posterior()

    with open(save_path, "wb") as handle:
        pickle.dump(amortized_posterior, handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_simulations", type=int, default=20000)
    parser.add_argument("--sigma", type=float, default=150.)
    args = parser.parse_args()
    main(args.sigma, args.num_simulations, args.save_path)