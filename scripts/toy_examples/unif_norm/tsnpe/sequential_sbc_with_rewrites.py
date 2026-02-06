import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator, get_all_test_function_names_list, get_test_function
from sbc.sbc_tools import sbc_ranks_tsnpe, train_tsnpe_posterior
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "tsnpe")


def main(sigma, N_iter, N_samp, num_sequential_rounds, num_simulations_per_round, 
         experiment_ID, d, L, U, test_function_name, sample_with, epsilon, restricted_prior_sample_with):
    if d > 1 and not test_function_name:
        raise AssertionError("TEST FUNCTION MUST BE SPECIFIED IF d > 1!")
    
    # By default, experiment_ID is -1, meaning we start a new experiment ID.
    continue_experiment = experiment_ID >= 0
    if not continue_experiment:
        # Find next available ID
        i = 0
        while os.path.exists(results_path + f"/sequential_sbc_ranks{i}.yaml"):
            i += 1
    else:
        # Continue with existing experiment
        i = experiment_ID
        if not os.path.exists(results_path + f"/sequential_sbc_ranks{i}.yaml"):
            raise AssertionError("No file in directory " + results_path + f"/sequential_sbc_ranks{i}.yaml")
        print(f"\n Continuing to append to experiment {experiment_ID}.")

    # Get simulator in required form
    simulator_ = lambda x: simulator(x, sigma=sigma, d=d)

    # Always save ranks as dict (even if singular / no test function)
    sequential_sbc_path = results_path + f"/sequential_sbc_ranks{i}.npz"
    config_path = results_path + f"/sequential_sbc_ranks{i}.yaml"
    simulations_path = results_path + f"/sequential_sbc_ranks{i}_simulations.npz"

    # Retrieve test function
    if test_function_name == "all":
        # Multiple test functions case
        print("Using all test functions")
        all_test_function_names = get_all_test_function_names_list(d=d)
        test_function = [(get_test_function(_test_function_name, d=d), _test_function_name) for _test_function_name in all_test_function_names]
    else:
        # Singular test function (or None)
        test_function = get_test_function(test_function_name, d=d)
    
    prior = make_prior(L=L, U=U, d=d)
    print("\n Running SBC:")
    for k in range(N_iter):
        print(f"\n SBC round {k} out of {N_iter}")
        print("\n Generating rank")

        start_time = time.perf_counter()
        try:
            if d == 1:
                if test_function_name:
                    raise NotImplementedError("YET TO IMPLEMENT 1D TEST FUNCTIONS!")
                rank_sequential, sample_dict = sbc_ranks_tsnpe(simulator_,
                                        prior,
                                        train_tsnpe_posterior,
                                        test_function=test_function,
                                        N_iter=1,
                                        N_samp=N_samp,
                                        num_sequential_rounds=num_sequential_rounds,
                                        num_simulations_per_round=num_simulations_per_round,
                                        show_progress=True,
                                        return_samples=True,
                                        always_return_dict=True,
                                        sample_with=sample_with,
                                        restricted_prior_sample_with=restricted_prior_sample_with,
                                        epsilon=epsilon)
            else:
                rank_sequential = sbc_ranks_tsnpe(simulator_,
                                        prior,
                                        train_tsnpe_posterior,
                                        test_function=test_function,
                                        N_iter=1,
                                        N_samp=N_samp,
                                        num_sequential_rounds=num_sequential_rounds,
                                        num_simulations_per_round=num_simulations_per_round,
                                        show_progress=True,
                                        return_samples=False,
                                        always_return_dict=True,
                                        sample_with=sample_with,
                                        restricted_prior_sample_with=restricted_prior_sample_with,
                                        epsilon=epsilon)
            end_time = time.perf_counter()
            print("\n Rank generated")
            sbc_time = end_time - start_time
        except AssertionError as e:
            print(f"\n SBC ROUND {k} FAILED! SKIPPING ROUND.")
            print(f"\n Assertion error: {e}")
            sbc_time = np.nan
            if test_function_name == "all":
                # make rank dict of np.nans
                rank = {test_function_name_: [np.nan] for test_function_name_ in all_test_function_names}
            else:
                rank = {"": [np.nan]}
            print("\n Rank generation failed.")
        
        if (k == 0) and (not continue_experiment):
            config = {"sigma": sigma, 
                "N_iter": 1,
                "N_samp": N_samp,
                "num_sequential_rounds": num_sequential_rounds,
                "num_simulations_per_round": num_simulations_per_round,
                "sbc_times": [sbc_time],
                "total_sbc_time": sbc_time,
                "d": d,
                "L": L,
                "U": U,
                "test_function_name": all_test_function_names if test_function_name=="all" else test_function_name,
                "sample_with": sample_with,
                "epsilon": epsilon, 
                "restricted_prior_sample_with": restricted_prior_sample_with
            }
            
            print(f"\n Saving first rank to {sequential_sbc_path}:")
            np.savez(sequential_sbc_path, **rank_sequential)
            print("\n Rank saved.")

            print(f"\n Saving first config file to {config_path}:")
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)
            print("\n Config file saved successfully.")

            if d == 1:
                # Save simulations:
                print(f"\n Saving first simulations to {simulations_path}:")
                np.savez(simulations_path, **sample_dict)
                print("\n Simulations saved successfully.")

        else:
            # Append to existing files

            # Load ranks
            sequential_ranks = np.load(sequential_sbc_path)
            sequential_ranks = dict(sequential_ranks)
            # Update ranks by appending new rank
            for test_function_name_ in sequential_ranks:
                sequential_ranks[test_function_name_] = list(sequential_ranks[test_function_name_])
                sequential_ranks[test_function_name_].append(rank_sequential[test_function_name_][0])
            # Save updated ranks
            np.savez(sequential_sbc_path, **sequential_ranks)

            # Load config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = dict(config)
            # Assert that hyperparameters are equal
            if k==0:
                # If we are appending to an existing experiment, we must assert that
                # all hyperparameters are equal. If not, it's not sensible to append simulations.
                assert(
                    config["N_samp"] == N_samp
                    and config["sigma"] == sigma
                    and config["num_sequential_rounds"] == num_sequential_rounds
                    and config["num_simulations_per_round"] == num_simulations_per_round
                    and config["d"] == d
                    and config["L"] == L
                    and config["U"] == U
                    and config["sample_with"] == sample_with
                    and config["epsilon"] == epsilon
                    and config["restricted_prior_sample_with"] == restricted_prior_sample_with
                )
                if test_function_name=="all":
                    assert config["test_function_name"] == all_test_function_names
                else:
                    assert config["test_function_name"] == test_function_name
            old_N_iter = config["N_iter"] # For simulation indexing
            # Update config
            config["N_iter"] += 1
            config["sbc_times"].append(sbc_time)
            if not np.isnan(sbc_time):
                config["total_sbc_time"] += sbc_time
            # Save updated config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

            if d == 1:
                # Load simulations
                samples_dict = np.load(simulations_path)
                samples_dict = dict(samples_dict)
                # Append new simulations
                samples_dict[f"prior_sample_round_{old_N_iter}"] = sample_dict["prior_sample_round_0"]
                samples_dict[f"data_sample_round_{old_N_iter}"] = sample_dict["data_sample_round_0"]
                samples_dict[f"posterior_samples_round_{old_N_iter}"] = sample_dict["posterior_samples_round_0"]
                # Save simulations
                np.savez(simulations_path, **samples_dict)
    print("\n SBC finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=1000)
    parser.add_argument("--N_samp", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--experiment_ID", type=int, default=-1) # By default, start new experiment
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--test_function_name", type=str, default=None)
    parser.add_argument("--sample_with", type=str, default="direct")
    parser.add_argument("--restricted_prior_sample_with", type=str, default="rejection")
    parser.add_argument("--epsilon", type=float, default=1e-4)


    args = parser.parse_args()
    main(args.sigma, args.N_iter, args.N_samp, args.num_sequential_rounds, args.num_simulations_per_round,
         args.experiment_ID, args.d, args.L, args.U, args.test_function_name, args.sample_with,
         args.epsilon, args.restricted_prior_sample_with)
