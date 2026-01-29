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
import time

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
        tau, N, ll, psi, bl, use_combined_loss, experiment_ID):
    
    # By default, experiment_ID is -1, meaning we start a new experiment ID.
    continue_experiment = experiment_ID >= 0
    if not continue_experiment:
        # Find next available ID
        i = 0
        while os.path.exists(results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"):
            i += 1
    else:
        # Continue with existing experiment
        i = experiment_ID
        if not os.path.exists(results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"):
            raise AssertionError("No file in directory " + results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml")
        print(f"\n Continuing to append to experiment {experiment_ID}.")

    config_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"
    if test_function_name=="all":
        # If using all test functions, need to save a dict of np arrays as ranks
        ranks_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".npz"
    else:
        # If using a single test function, we simply save a single numpy array
        ranks_path = results_path + f"/sequential_sbc{i}_leader_trajectory{leader_trajectory_ID}" + ".npy"
    
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
    
    leader_trajectory_name = f"leader_trajectory{leader_trajectory_ID}"
    path_to_leader_trajectory = trajectories_path + "/" + leader_trajectory_name + ".npz"
    path_to_leader_trajectory_config = trajectories_path + "/" + leader_trajectory_name + ".yaml"
    # Retrieve leader trajectory config
    with open(path_to_leader_trajectory_config, "r") as f:
        leader_trajectory_config = yaml.safe_load(f) # Dictionary
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

    print("\n Running SBC:")
    for n in range(N_iter):
        print(f"\n SBC round {n} out of {N_iter}")
        print("\n Generating rank")
        start_time = time.perf_counter()
        rank = sbc_ranks_snpe_c(simulator_,
                        prior,
                        train_snpe_c_posterior,
                        test_function=test_function,
                        N_iter=1,
                        N_samp=N_samp,
                        num_sequential_rounds=num_sequential_rounds,
                        num_simulations_per_round=num_simulations_per_round,
                        use_combined_loss=use_combined_loss, # Using combined loss can help reduce leakage in models with compact prior supports
                        show_progress=False)
        print("rank:")
        print(rank) ########
        end_time = time.perf_counter()
        print("\n Rank generated")
        sbc_time = end_time - start_time

        if (n == 0) and (not continue_experiment):
            sbc_config = {
                    "N_iter": 1,
                    "N_samp": N_samp,
                    "test_function_name": all_test_function_names if test_function_name=="all" else test_function_name,
                    "num_sequential_rounds": num_sequential_rounds,
                    "num_simulations_per_round": num_simulations_per_round,
                    "tau": tau, 
                    "N": N,
                    "ll": ll, 
                    "psi": psi,
                    "bl": bl, 
                    "leader_trajectory_ID": leader_trajectory_ID,
                    "sbc_times": [sbc_time],
                    "total_sbc_time": sbc_time
                }

            config = {
                "leader_trajectory_config": leader_trajectory_config,
                "sbc_config": sbc_config,
                "prior_config": prior_config
            }

            if isinstance(rank, dict):
                print(f"Saving first rank to {ranks_path}:")
                np.savez(ranks_path, **rank)
                print("Rank saved successfully.")
            else:
                print(f"Saving first rank to {ranks_path}:")
                np.save(ranks_path, np.array(rank).reshape(-1))
                print("Rank saved successfully.")

            print(f"Saving first config file to path {config_path}:")
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)
            print("Config file saved successfully.")
        else:
            # Load ranks
            ranks = np.load(ranks_path)
            if isinstance(rank, dict):
                ranks = dict(ranks)
                print("ranks:")
                print(ranks)
                # Update ranks by appending new rank
                for test_function_name_ in ranks:
                    ranks[test_function_name_] = list(ranks[test_function_name_])
                    ranks[test_function_name_].append(rank[test_function_name_][0])
                # Save updated ranks
                np.savez(ranks_path, **ranks)
            else:
                # Update ranks by appending new rank
                ranks = list(ranks)
                ranks.append(rank[0])
                # Save updated ranks
                np.save(ranks_path, np.array(ranks))

            # Load config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = dict(config)
            # Assert that hyperparameters are equal
            if n==0:
                # If we are appending to an existing experiment, we must assert that
                # all hyperparameters are equal. If not, it's not sensible to append simulations.

                assert(
                    config["sbc_config"]["N_samp"] == N_samp
                    and config["sbc_config"]["num_sequential_rounds"] == num_sequential_rounds
                    and config["sbc_config"]["num_simulations_per_round"] == num_simulations_per_round
                    and config["sbc_config"]["leader_trajectory_ID"] == leader_trajectory_ID
                    and config["sbc_config"]["tau"] == tau
                    and config["sbc_config"]["N"] == N
                    and config["sbc_config"]["ll"] == ll
                    and config["sbc_config"]["psi"] == psi
                    and config["sbc_config"]["bl"] == bl
                    and config["leader_trajectory_config"] == leader_trajectory_config
                    and config["prior_config"] == prior_config
                )
                if test_function_name=="all":
                    assert config["sbc_config"]["test_function_name"] == all_test_function_names
                else:
                    assert config["sbc_config"]["test_function_name"] == test_function_name

            old_N_iter = config["sbc_config"]["N_iter"] # For simulation indexing
            # Update config
            config["sbc_config"]["N_iter"] += 1
            config["sbc_config"]["sbc_times"].append(sbc_time)
            config["sbc_config"]["total_sbc_time"] += sbc_time
            # Save updated config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

    print("\n SBC finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--test_function_name", type=str, default="all")
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)

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
    parser.add_argument("--leader_trajectory_ID", type=int, required=True)
    parser.add_argument("--use_combined_loss", type=bool, default=True)
    parser.add_argument("--experiment_ID", type=int, default=-1)

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
        args.tau, args.N, args.ll, args.psi, args.bl, args.use_combined_loss, args.experiment_ID)
