import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.inference import NPE_C
import scipy
from torch.distributions import Exponential, Normal, InverseGamma, MultivariateNormal
from sbi.utils import BoxUniform, MultipleIndependent
from examples.gipps import make_prior_7d_npe_c, simulator, get_test_function, all_test_function_names
from sbc.sbc_tools import sbc_ranks_snpe_c, train_snpe_c_posterior
import argparse
from pathlib import Path
import pickle
import yaml
import os

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "npe_c")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "trajectories")


def main(N_iter, N_samp, num_sequential_rounds, num_simulations_per_round,
        leader_trajectory_ID, test_function_name,
        aL, aU,
        bL, bU,
        VL, VU,
        xf0L, xf0U,
        vf0L, vf0U,
        prior_mean_mu, prior_variance_mu,
        prior_alpha_sigmasquared, prior_beta_sigmasquared,
        tau, N, ll, psi, bl):
    
    prior_config = {
            "aL": aL,
            "aU": aU,
            "bL": bL,
            "bU": bU,
            "VL": VL,
            "VU": VU,
            "xf0L": xf0L,
            "xf0U": xf0U,
            "vf0L": vf0L,
            "vf0U": vf0U,
            "prior_mean_mu": prior_mean_mu,
            "prior_variance_mu": prior_variance_mu,
            "prior_alpha_sigmasquared": prior_alpha_sigmasquared,
            "prior_beta_sigmasquared": prior_beta_sigmasquared,
        }
    
    prior = make_prior_7d_npe_c(aL, aU,
                        bL, bU,
                        VL, VU,
                        xf0L, xf0U,
                        vf0L, vf0U,
                        prior_mean_mu, prior_variance_mu,
                        prior_alpha_sigmasquared, prior_beta_sigmasquared)

    sbc_config = {
        "N_iter": N_iter,
        "N_samp": N_samp,
        "test_function_name": all_test_function_names if test_function_name=="all" else test_function_name,
        "num_sequential_rounds": num_sequential_rounds,
        "num_simulations_per_round": num_simulations_per_round,
        "tau": tau, 
        "N": N,
        "ll": ll, 
        "psi": psi,
        "bl": bl, 
        "leader_trajectory_ID": leader_trajectory_ID
    }

    leader_trajectory_name = f"leader_trajectory{leader_trajectory_ID}"
    path_to_leader_trajectory = trajectories_path + "/" + leader_trajectory_name + ".npz"
    path_to_leader_trajectory_config = trajectories_path + "/" + leader_trajectory_name + ".yaml"

    # Retrieve leader trajectory config
    with open(path_to_leader_trajectory_config, "r") as f:
        leader_trajectory_config = yaml.safe_load(f) # Dictionary

    config = {
        "leader_trajectory_config": leader_trajectory_config,
        "sbc_config": sbc_config,
        "prior_config": prior_config
    }
    
    # Retrieve xl, vl
    leader_trajectory = np.load(path_to_leader_trajectory)
    xl = leader_trajectory["xl"]
    vl = leader_trajectory["vl"]
    
    # Retrieve test function
    if test_function_name=="all":
        print("Using all test functions")
        test_function = [(get_test_function(_test_function_name), _test_function_name) for _test_function_name in all_test_function_names]
    else:
        print("Using test function:" + test_function_name)
        test_function = get_test_function(test_function_name)

    # Ensure simulator is in correct format
    simulator_ = lambda x: simulator(x, tau, N, ll, psi, xl, vl, bl)

    # Generate ranks
    print("Generating ranks:")
    # Ranks will be 1) a np array (if single test function) or 2) a dict of np arrays (if test_function_name is "all")
    ranks = sbc_ranks_snpe_c(simulator_,
                    prior,
                    train_snpe_c_posterior,
                    test_function=test_function,
                    N_iter=N_iter,
                    N_samp=N_samp,
                    num_sequential_rounds=num_sequential_rounds,
                    num_simulations_per_round=num_simulations_per_round,
                    show_progress=True)
    print("Ranks generated successfully.")

    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}.npy") \
        or os.path.exists(results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}.npz"):
        i += 1

    if isinstance(ranks, dict):
        sequential_sbc_save_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".npz"
        print(f"Saving ranks to {sequential_sbc_save_path}:")
        np.savez(sequential_sbc_save_path, **ranks)
        print("Ranks saved successfully.")
    else:
        sequential_sbc_save_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".npy"
        print(f"Saving ranks to {sequential_sbc_save_path}:")
        np.save(sequential_sbc_save_path, ranks)
        print("Ranks saved successfully.")

    config_save_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"
    print(f"Saving config file to path {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("Config file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--test_function_name", type=str, default="all")
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--leader_trajectory_ID", type=int, required=True)

    parser.add_argument("--aL", type=float, default=0.5)
    parser.add_argument("--aU", type=float, default=3.5)
    parser.add_argument("--bL", type=float, default=-6.)
    parser.add_argument("--bU", type=float, default=-1.)
    parser.add_argument("--VL", type=float, default=15.)
    parser.add_argument("--VU", type=float, default=35.)
    parser.add_argument("--xf0L", type=float, default=-60.)
    parser.add_argument("--xf0U", type=float, default=-10.)
    parser.add_argument("--vf0L", type=float, default=5.)
    parser.add_argument("--vf0U", type=float, default=25.)
    parser.add_argument("--prior_mean_mu", type=float, default=0.) 
    parser.add_argument("--prior_variance_mu", type=float, default=9.)
    parser.add_argument("--prior_alpha_sigmasquared", type=float, default=1.)
    parser.add_argument("--prior_beta_sigmasquared", type=float, default=3.)

    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--ll", type=float, default=7.5)
    parser.add_argument("--psi", type=float, default=1.05)
    parser.add_argument("--bl", type=float, default=-4.)

    args = parser.parse_args()
    main(args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round,
        args.leader_trajectory_ID, args.test_function_name,
        args.aL, args.aU,
        args.bL, args.bU,
        args.VL, args.VU,
        args.xf0L, args.xf0U,
        args.vf0L, args.vf0U,
        args.prior_mean_mu, args.prior_variance_mu,
        args.prior_alpha_sigmasquared, args.prior_beta_sigmasquared,
        args.tau, args.N, args.ll, args.psi, args.bl)
