import torch
from sbi.inference import NPE_A
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
from sbc.sbc_tools import sbc_ranks
import argparse

def main(save_path, sigma, posterior_name, N_iter, N_samp):
    prior = make_prior(sigma)
    amortized_posterior_path = "/Users/Lieve/Documents/Masters Project/SBC-SBI/results/toy_examples/norm_norm_diffuse_1d/npe_a/" + posterior_name + ".pkl"
    with open(amortized_posterior_path, "rb") as f:
        amortized_posterior = pickle.load(f)
    print("Running SBC:")
    ranks = np.array(sbc_ranks(model=simulator, prior=prior, posterior=amortized_posterior, N_iter=N_iter, N_samp=N_samp))
    print("SBC finished.")
    save_path = save_path + f"/amortized_sbc_ranks_N_iter{N_iter}_N_samp{N_samp}_sigma{sigma}_{posterior_name}.npy"
    print("Saving ranks:")
    np.save(save_path, ranks)
    print("Ranks saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--posterior_name", type=str, required=True)
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=150.)
    args = parser.parse_args()
    main(args.save_path, args.sigma, args.posterior_name, args.N_iter, args.N_samp)