import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator, get_test_function, get_all_test_function_names_list
from sbc.sbc_tools import sbc_ranks
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "tsnpe")


def main(N_iter, N_samp, sigma, amortized_posterior_ID, d, L, U, test_function_name):
    if d > 1 and not test_function_name:
        raise AssertionError("TEST FUNCTION MUST BE SPECIFIED IF d > 1!")

    prior = make_prior(L=L, U=U, d=d)
    amortized_posterior_path = results_path + f"/amortized_posterior{amortized_posterior_ID}.pkl"
    # Retrieve amortized posterior
    with open(amortized_posterior_path, "rb") as f:
        amortized_posterior = pickle.load(f)
    print("\n Running SBC:")
    start_time = time.perf_counter()
    # Get simulator in required form
    simulator_ = lambda x: simulator(x, sigma=sigma, d=d)

    # Retrieve test function
    if test_function_name == "all":
        # Multiple test functions case
        print("Using all test functions")
        all_test_function_names = get_all_test_function_names_list(d=d)
        test_function = [(get_test_function(_test_function_name, d=d), _test_function_name) for _test_function_name in all_test_function_names]
    else:
        # Singular test function (or None)
        test_function = get_test_function(test_function_name, d=d)

    if d == 1:
        if test_function_name:
            raise NotImplementedError("YET TO IMPLEMENT 1D TEST FUNCTIONS!")
        # Only save samples if d=1
        # Ranks will be a dict either way
        ranks, samples_dict = sbc_ranks(model=simulator_, prior=prior, posterior=amortized_posterior, N_iter=N_iter, N_samp=N_samp, show_progress=False, return_samples=True, test_function=test_function, always_return_dict=True)   
    else:
        # Ranks will be a dict either way
        ranks = sbc_ranks(model=simulator_, prior=prior, posterior=amortized_posterior, N_iter=N_iter, N_samp=N_samp, show_progress=False, return_samples=False, test_function=test_function, always_return_dict=True)

    end_time = time.perf_counter()
    print("\n SBC finished.")
    total_sbc_time = end_time - start_time

    config = {"N_iter": N_iter,
              "N_samp": N_samp,
              "sigma": sigma,
              "amortized_posterior_ID": amortized_posterior_ID,
              "total_sbc_time": total_sbc_time,
              "d": d,
              "L": L,
              "U": U,
              "test_function_name": get_all_test_function_names_list(d=d) if test_function_name=="all" else test_function_name}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.yaml"):
        i += 1

    sbc_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.npz"
    config_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}.yaml"
    if d == 1:
        simulations_save_path = results_path + f"/amortized_sbc_ranks{i}_amortized_posterior{amortized_posterior_ID}_simulations.npz"
    
    print(f"\n Saving ranks to {sbc_save_path}:")
    np.savez(sbc_save_path, **ranks)
    print("\n Ranks saved.")

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    # Save simulations if d=1:
    if d == 1:
        print(f"\n Saving simulations to {simulations_save_path}:")
        np.savez(simulations_save_path, **samples_dict)
        print("\n Simulations saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amortized_posterior_ID", type=int, required=True)
    parser.add_argument("--N_iter", type=int, default=1000)
    parser.add_argument("--N_samp", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--test_function_name", type=str, default=None)
    args = parser.parse_args()
    main(args.N_iter, args.N_samp, args.sigma, args.amortized_posterior_ID,
         args.d, args.L, args.U, args.test_function_name)
