import torch
from sbi.inference import NPE_A
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
from sbc.sbc_tools import sbc_ranks
import argparse
from pathlib import Path

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_a")

def main(N_iter, N_samp, sigma, posterior_name):
    prior = make_prior(sigma)
    amortized_posterior_path = results_path + "/" + posterior_name + ".pkl"
    with open(amortized_posterior_path, "rb") as f:
        amortized_posterior = pickle.load(f)
    print("Running SBC:")
    ranks = np.array(sbc_ranks(model=simulator, prior=prior, posterior=amortized_posterior, N_iter=N_iter, N_samp=N_samp, show_progress=True))
    print("SBC finished.")
    save_path = results_path + f"/amortized_sbc_ranks_Niter{N_iter}_Nsamp{N_samp}_sig{sigma}_{posterior_name}.npy"
    print(f"Saving ranks to {save_path}:")
    np.save(save_path, ranks)
    print("Ranks saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior_name", type=str, required=True)
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=150.)
    args = parser.parse_args()
    main(args.N_iter, args.N_samp, args.sigma, args.posterior_name)