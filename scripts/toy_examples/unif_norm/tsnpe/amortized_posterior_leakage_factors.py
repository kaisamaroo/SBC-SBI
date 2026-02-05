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


def train_amortized_posterior_density_estimator(sigma, num_simulations, d, L, U, force_first_round_loss):
    prior = make_prior(L=L, U=U, d=d)
    inference = NPE_C(prior=prior)
    print("\n Generating samples:")
    parameter_samples = prior.sample((num_simulations,))  # shape (num_simulations, d)
    data_samples = simulator(parameter_samples, sigma=sigma, d=d)  # shape (num_simulations, nd)
    print("\n Samples generated successfully.")
    print("\n Training density estimator:")
    inference = inference.append_simulations(parameter_samples, data_samples)
    density_estimator = inference.train(force_first_round_loss=force_first_round_loss)
    print("\n Density estimator trained successfully.")
    return density_estimator 


def main(sigma, num_simulations, d, L, U, num_simulations_list, d_list, force_first_round_loss,
         num_posteriors_per_leakage_factor, num_posterior_samples_per_leakage_factor,
         x_observed):
    
    config = {
        "sigma": sigma,
        "num_simulations": num_simulations,
        "d": d,
        "L": L,
        "U": U,
        "num_simulations_list": num_simulations_list,
        "d_list": d_list,
        "force_first_round_loss": force_first_round_loss,
        "num_posteriors_per_leakage_factor": num_posteriors_per_leakage_factor,
        "num_posterior_samples_per_leakage_factor": num_posterior_samples_per_leakage_factor,
        "x_observed": x_observed,
    }

    leakage_factor_dict = {}

    if isinstance(num_simulations_list, list) and isinstance(d_list, list):
        raise ValueError("EXACTLY ONE OF num_simulations_list AND d_list HAS TO BE INPUT! Currently, both were input.")
    
    elif isinstance(num_simulations_list, list):
        # Force user to pass x_observed if they are varying num_simulations (don't allow if d is being varies)
        if not x_observed:
            # FUTURE: Could implement so that this samples from prior if x_observed isnt input
            raise AssertionError("x_observed MUST BE INPUT IF num_simulations IS BEING VARIED!")
        # x_observed will be a list (singleton [x] if d=1). Must have length d
        if not len(x_observed) == d:
            raise AssertionError(f"x_observed MUST HAVE LENGTH d, CURRENTLY HAS LENGTH {len(x_observed)} AND d IS {d}")
        
        x_observed = torch.tensor(x_observed).unsqueeze(0) # shape torch.size([1, d]) needed for sampling from density_estimator

        for num_simulations in num_simulations_list:
            print(f"\n num_simulations = {num_simulations}:")
            leakage_factors = []
            for _ in range(num_posteriors_per_leakage_factor):
                print(f"\n Posterior {_+1} out of {num_posteriors_per_leakage_factor}")
                density_estimator = train_amortized_posterior_density_estimator(sigma, num_simulations, d, L, U, force_first_round_loss)
                samples = density_estimator.sample((num_posterior_samples_per_leakage_factor,),
                                                    condition=x_observed).detach() # shape torch.size([num_posterior_samples_per_leakage_factor, 1, d])
                samples = samples.squeeze(1) # shape torch.size([num_posterior_samples_per_leakage_factor, d])
                leaked_mask = (samples < L) | (samples > U)
                leakage_factor = leaked_mask.any(dim=1).sum().item() / num_posterior_samples_per_leakage_factor
                leakage_factors.append(leakage_factor)
            leakage_factor_dict[f"num_simulations = {num_simulations}"] = leakage_factors
                
    elif isinstance(d_list, list):
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
                density_estimator = train_amortized_posterior_density_estimator(sigma, num_simulations, d, L, U, force_first_round_loss)
                samples = density_estimator.sample((num_posterior_samples_per_leakage_factor,),
                                                    condition=x_observed).detach() # shape torch.size([num_posterior_samples_per_leakage_factor, 1, d])
                samples = samples.squeeze(1) # shape torch.size([num_posterior_samples_per_leakage_factor, d])
                leaked_mask = (samples < L) | (samples > U)
                leakage_factor = leaked_mask.any(dim=1).sum().item() / num_posterior_samples_per_leakage_factor
                leakage_factors.append(leakage_factor)
            leakage_factor_dict[f"d = {d}"] = leakage_factors

    else:
        raise ValueError("EXACTLY ONE OF num_simulations_list AND d_list HAS TO BE INPUT! Currently, none were input.")

    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_leakage_factor_dict{i}.npz"):
        i += 1

    # Define save paths
    amortized_leakage_factor_dict_save_path = results_path + f"/amortized_leakage_factor_dict{i}.npz"
    config_save_path = results_path + f"/amortized_leakage_factor_dict{i}.yaml"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving leakage factor dict to {amortized_leakage_factor_dict_save_path}:")
    np.savez(amortized_leakage_factor_dict_save_path, **leakage_factor_dict)
    print("\n Saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Exactly one of these must be input
    parser.add_argument("--num_simulations_list", type=int, nargs="+", default=None)
    parser.add_argument("--d_list", type=int, nargs="+", default=None)

    parser.add_argument("--x_observed", type=float, nargs="+", default=None)
    parser.add_argument("--num_simulations", type=int, default=20000)
    parser.add_argument("--num_posteriors_per_leakage_factor", type=int, default=100)
    parser.add_argument("--num_posterior_samples_per_leakage_factor", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--force_first_round_loss", type=bool, default=False)
    
    args = parser.parse_args()
    main(args.sigma, args.num_simulations, args.d, args.L, args.U, args.num_simulations_list,
         args.d_list, args.force_first_round_loss, args.num_posteriors_per_leakage_factor,
         args.num_posterior_samples_per_leakage_factor, args.x_observed)