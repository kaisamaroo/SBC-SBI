import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.norm_norm_diffuse_1d import make_prior, simulator
from sbc.sbc_tools import sbc_ranks_snpe_c, sbc_ranks
import argparse
from pathlib import Path

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "norm_norm_diffuse_1d" / "npe_c")


def train_sequential_posterior(simulator, prior, x_observed, num_sequential_rounds, num_simulations_per_round):
    inference = NPE_C(prior)
    proposal = prior
    for r in range(num_sequential_rounds):
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)

        if r == num_sequential_rounds - 1:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train()
            sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train()
            sequential_posterior = inference.build_posterior().set_default_x(x_observed)
            proposal = sequential_posterior
    return sequential_posterior


def main(sigma, N_iter, N_samp, num_sequential_rounds, num_simulations_per_round, show_progress):
    prior = make_prior(sigma)
    print("Running SBC:")
    ranks_sequential = sbc_ranks_snpe_c(simulator,
                            prior,
                            train_sequential_posterior,
                            test_function=None,
                            N_iter=N_iter,
                            N_samp=N_samp,
                            num_sequential_rounds=num_sequential_rounds,
                            num_simulations_per_round=num_simulations_per_round,
                            show_progress=show_progress)
    ranks_sequential = np.array(ranks_sequential)
    print("SBC finished.")
    save_path = results_path + f"/sequential_sbc_ranks_Niter{N_iter}_Nsamp{N_samp}_sig{sigma}_nspr{num_simulations_per_round}_nr{num_sequential_rounds}.npy"
    print(f"Saving ranks to {save_path}:")
    np.save(save_path, ranks_sequential)
    print("Ranks saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=150.)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--show_progress", type=bool, default=True)
    args = parser.parse_args()
    main(args.sigma, args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round, args.show_progress)
