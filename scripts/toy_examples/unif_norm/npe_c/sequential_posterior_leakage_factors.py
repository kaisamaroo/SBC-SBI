import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
import argparse
from pathlib import Path
import os
import time
import yaml

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "npe_c")


def train_sequential_posterior_density_estimator(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss):
    prior = make_prior(L=L, U=U, d=d)
    inference = NPE_C(prior=prior)
    proposal = prior

    for r in range(num_sequential_rounds):
        print(f"\n Round {r+1}:")
        print("\n Generating samples:")
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples, sigma=sigma, d=d)
        print("\n Samples generated")

        print("\n Training proposal:")
        if r == num_sequential_rounds - 1:
            density_estimator = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            sequential_posterior = inference.build_posterior().set_default_x(x_observed)
            proposal = sequential_posterior
        print("\n Proposal trained successfully:")
    print("\n Posterior trained successfully.")
    return density_estimator


def main(sigma, d, L, U, num_sequential_rounds_list, num_simulations_per_round_list, 
         d_list, use_combined_loss, num_posteriors_per_leakage_factor, 
         num_posterior_samples_per_leakage_factor, x_observed, num_sequential_rounds, num_simulations_per_round):
    
    config = {
        "sigma": sigma,
        "d": d,
        "L": L,
        "U": U,
        "num_sequential_rounds_list": num_sequential_rounds_list,
        "num_simulations_per_round_list": num_simulations_per_round_list,
        "d_list": d_list,
        "use_combined_loss": use_combined_loss,
        "num_sequential_rounds": num_sequential_rounds,
        "num_simulations_per_round": num_simulations_per_round,
        "num_posteriors_per_leakage_factor": num_posteriors_per_leakage_factor,
        "num_posterior_samples_per_leakage_factor": num_posterior_samples_per_leakage_factor,
        "x_observed": x_observed,
    }

    leakage_factor_dict = {}

    both_nr_nspr_are_lists = isinstance(num_sequential_rounds_list, list) and isinstance(num_simulations_per_round_list, list)
    d_is_list = isinstance(d_list, list)

    if both_nr_nspr_are_lists and d_is_list:
        raise ValueError("INVALID INPUT TYPES: EITHER BOTH num_sequential_rounds_list AND num_simulations_per_round_list MUST BE LISTS, XOR d_list MUST BE A LIST! Currently, both are true!")
    elif both_nr_nspr_are_lists:
        # Force user to pass x_observed if they are varying num_sequential_rounds and num_simulations_per_round (don't allow if d is being varies)
        if not x_observed:
            # FUTURE: Could implement so that this samples from prior if x_observed isnt input
            raise AssertionError("x_observed MUST BE INPUT IF num_sequential_rounds and num_simulations_per_round ARE BEING VARIED!")
        # x_observed will be a list (singleton [x] if d=1). Must have length d
        if not len(x_observed) == d:
            raise AssertionError(f"x_observed MUST HAVE LENGTH d, CURRENTLY HAS LENGTH {len(x_observed)} AND d IS {d}")
        
        x_observed = torch.tensor(x_observed).unsqueeze(0) # shape torch.size([1, d]) needed for sampling from density_estimator

        for num_sequential_rounds, num_simulations_per_round in zip(num_sequential_rounds_list, num_simulations_per_round_list):
            print(f"(num_sequential_rounds, num_simulations_per_round) = {(num_sequential_rounds, num_simulations_per_round)}:")
            leakage_factors = []
            for _ in range(num_posteriors_per_leakage_factor):
                density_estimator = train_sequential_posterior_density_estimator(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss)
                samples = density_estimator.sample((num_posterior_samples_per_leakage_factor,),
                                                    condition=x_observed).detach() # shape torch.size([num_posterior_samples_per_leakage_factor, 1, d])
                samples = samples.squeeze(1) # shape torch.size([num_posterior_samples_per_leakage_factor, d])
                leaked_mask = (samples < L) | (samples > U)
                leakage_factor = leaked_mask.any(dim=1).sum().item() / num_posterior_samples_per_leakage_factor
                leakage_factors.append(leakage_factor)
            leakage_factor_dict[f"(num_sequential_rounds, num_simulations_per_round) = {(num_sequential_rounds, num_simulations_per_round)}"] = leakage_factors
        
    elif d_is_list:
        # Cannot specify x_observed if d is being varied
        if x_observed:
            raise AssertionError("CANNOT INPUT x_observed IF d IS BEING VARIED!")
        for d in d_list:
            print(f"\n d = {d}:")
            # sample conditioner from prior predictive
            prior = make_prior(L=L, U=U, d=d)
            true_mu = prior.sample() # shape torch.size([d])
            x_observed = simulator(true_mu, sigma=sigma, d=d) # shape torch.size([d])
            x_observed = x_observed.unsqueeze(0) # shape torch.size([1, d]) needed for sampling from density_estimator
            print(f"x_observed = {x_observed}")
            leakage_factors = []
            for _ in range(num_posteriors_per_leakage_factor):
                print(f"\n Posterior {_+1} out of {num_posteriors_per_leakage_factor}")
                density_estimator = train_sequential_posterior_density_estimator(sigma, x_observed, num_sequential_rounds, num_simulations_per_round, d, L, U, use_combined_loss)
                samples = density_estimator.sample((num_posterior_samples_per_leakage_factor,),
                                                    condition=x_observed).detach() # shape torch.size([num_posterior_samples_per_leakage_factor, 1, d])
                samples = samples.squeeze(1) # shape torch.size([num_posterior_samples_per_leakage_factor, d])
                leaked_mask = (samples < L) | (samples > U)
                leakage_factor = leaked_mask.any(dim=1).sum().item() / num_posterior_samples_per_leakage_factor
                leakage_factors.append(leakage_factor)
            leakage_factor_dict[f"d = {d}"] = leakage_factors

    else:
        raise ValueError("INVALID INPUT TYPES: EITHER BOTH num_sequential_rounds_list AND num_simulations_per_round_list MUST BE LISTS, XOR d_list MUST BE A LIST! Currently, neither are true!")
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/sequential_leakage_factor_dict{i}.npz"):
        i += 1

    # Define save paths
    sequential_leakage_factor_dict_save_path = results_path + f"/sequential_leakage_factor_dict{i}.npz"
    config_save_path = results_path + f"/sequential_leakage_factor_dict{i}.yaml"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving leakage factor dict to {sequential_leakage_factor_dict_save_path}:")
    np.savez(sequential_leakage_factor_dict_save_path, **leakage_factor_dict)
    print("\n Saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_list", type=int, nargs="+", default=None)
    parser.add_argument("--num_sequential_rounds_list", type=int, nargs="+", default=None)
    parser.add_argument("--num_simulations_per_round_list", type=int, nargs="+", default=None)
    
    parser.add_argument("--x_observed", type=float, nargs="+", default=None)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--num_posteriors_per_leakage_factor", type=int, default=100)
    parser.add_argument("--num_posterior_samples_per_leakage_factor", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--use_combined_loss", type=bool, default=True)
    
    args = parser.parse_args()
    main(args.sigma, args.d, args.L, args.U, args.num_sequential_rounds_list, args.num_simulations_per_round_list, 
         args.d_list, args.use_combined_loss, args.num_posteriors_per_leakage_factor, 
         args.num_posterior_samples_per_leakage_factor, args.x_observed, args.num_sequential_rounds, args.num_simulations_per_round)