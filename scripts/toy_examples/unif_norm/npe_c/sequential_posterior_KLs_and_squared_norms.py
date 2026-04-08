import torch
from sbi.inference import NPE_C
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time
import math

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "npe_c")


def true_posterior_log_p(theta, x, sigma, d):
    """
    ASSUMES L IS LARGE ENOUGH SO THAT THE TRUNCNORM IS APPROXIMATELY NORM!
    Computes log density of N(x, sigma^2 I_d) evaluated at each row of theta.
    theta must be a Tensor of shape (batch_size, d)
    x must be a Tensor of shape (d,)
    Returns a Tensor of shape (batch_size,) consisting of the log probs
    """
    diff = theta - x  # broadcasting, shape (batch_size, d)
    sq_norm = (diff ** 2).sum(dim=1) # shape (batch_size, )
    log_norm_const = -0.5 * d * np.log(2 * torch.pi) - d * np.log(sigma)
    log_prob = log_norm_const - 0.5 * sq_norm / (sigma ** 2) # shape (batch_size, )
    return log_prob


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round,
         d, L, U, use_combined_loss, density_estimator, num_repetitions, save_posteriors):
    
    if save_posteriors:
        posteriors_dict = {f"repetition_{r}": {} for r in range(num_repetitions)}

    # Store each KL as a list (each list has length num_repetitions)
    KLs_dict = {f"round_{r}": np.zeros(num_repetitions) for r in range(num_sequential_rounds)}
    # Store each squared_norms as a row of a np array with num_repetitions rows
    squared_norms_dict = {f"round_{r}": np.zeros((num_repetitions, num_simulations_per_round)) for r in range(num_sequential_rounds)}
    
    # Initialize simulation and train time lists (averaged over num_repetitions)
    avg_simulation_times = {f"round_{r}": 0 for r in range(num_sequential_rounds)}
    avg_training_times = {f"round_{r}": 0 for r in range(num_sequential_rounds)}

    for rep in range(num_repetitions):
        print(f"\n Repetition {rep}:")
        prior = make_prior(L=L, U=U, d=d)
        inference = NPE_C(prior=prior, density_estimator=density_estimator)
        proposal = prior

        for r in range(num_sequential_rounds):
            print(f"\n Round {r+1}:")
            print("\n Generating samples:")
            sample_start_time = time.perf_counter()
            parameter_samples = proposal.sample((num_simulations_per_round,))
            data_samples = simulator(parameter_samples, sigma=sigma, d=d)
            sample_end_time = time.perf_counter()
            sample_time = sample_end_time - sample_start_time
            avg_simulation_times[f"round_{r}"] += sample_time / num_repetitions
            print("\n Samples generated")

            # Don't compute squared norms and KLs for the prior
            if r > 0:
                print("\n Computing squared norms:")
                # Save squared norms of parameter samples 
                squared_norms = (parameter_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
                squared_norms_dict[f"round_{r-1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
                print("\n Squared norms generated")

                print("\n Computing KLs:")
                # Save approximate (reverse) KL divergence:
                # Compute log density of density estimator
                log_q = proposal.log_prob(parameter_samples) # Tensor of shape (num_simulations_per_round, ). Note set_default_x has already been called
                # Compute log density of true posterior
                log_p = true_posterior_log_p(parameter_samples, torch.tensor(x_observed), sigma, d) # Tensor of shape (num_simulations_per_round, )
                KL = torch.mean(log_q - log_p).item() # Monte carlo approximation to reverse KL
                KLs_dict[f"round_{r-1}"][rep] = KL
                print("\n KLs generated")

            print("\n Training proposal:")
            training_start_time = time.perf_counter()
            if r == num_sequential_rounds - 1:
                density_estimator_ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
                sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
                if save_posteriors:
                    posteriors_dict[f"repetition_{rep}"][f"round_{r}"] = sequential_posterior
            else:
                _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
                sequential_posterior = inference.build_posterior().set_default_x(torch.tensor(x_observed))
                if save_posteriors:
                    posteriors_dict[f"repetition_{rep}"][f"round_{r}"] = sequential_posterior
                proposal = sequential_posterior
            training_end_time = time.perf_counter()
            training_time = training_end_time - training_start_time
            avg_training_times[f"round_{r}"] += training_time / num_repetitions
            print("\n Proposal trained successfully:")
        print("\n Posterior trained successfully.")

        # Append final posterior squared norm and KL
        parameter_samples = sequential_posterior.sample((num_simulations_per_round,), x=torch.tensor(x_observed))

        print("\n Computing final round squared norms:")
        # Save squared norms of parameter samples 
        squared_norms = (parameter_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
        squared_norms_dict[f"round_{num_sequential_rounds - 1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
        print("\n Squared norms generated")

        print("\n Computing final round KLs:")
        # Save approximate (reverse) KL divergence:
        # Compute log density of density estimator
        log_q = sequential_posterior.log_prob(parameter_samples, x=torch.tensor(x_observed)) # Tensor of shape (num_simulations_per_round, ). Note set_default_x has already been called
        # Compute log density of true posterior
        log_p = true_posterior_log_p(parameter_samples, torch.tensor(x_observed), sigma, d) # Tensor of shape (num_simulations_per_round, )
        KL = torch.mean(log_q - log_p).item() # Monte carlo approximation to reverse KL
        KLs_dict[f"round_{num_sequential_rounds - 1}"][rep] = KL
        print("\n KLs generated")


    config = {"sigma": sigma, 
              "x_observed": x_observed,
              "num_sequential_rounds": num_sequential_rounds,
              "num_simulations_per_round": num_simulations_per_round,
              "avg_simulation_times": avg_simulation_times,
              "avg_training_times": avg_training_times,
              "d": d,
              "L": L,
              "U": U,
              "use_combined_loss": use_combined_loss,
              "density_estimator": density_estimator,
              "num_repetitions": num_repetitions,
              "save_posteriors": save_posteriors}
    
    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/KLs_squared_norms{i}.yaml"):
        i += 1
    
    # Save paths
    config_save_path = results_path + f"/KLs_squared_norms{i}.yaml"
    KLs_save_path = results_path + f"/KLs_squared_norms{i}_KLs_dict.npz"
    squared_norms_save_path = results_path + f"/KLs_squared_norms{i}_squared_norms_dict.npz"
    if save_posteriors:
        posteriors_dict_save_path = results_path + f"/KLs_squared_norms{i}_posteriors_dict.pkl"

    print(f"\n Saving config file to {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("\n Config file saved successfully.")

    print(f"\n Saving KLs to {KLs_save_path}:")
    np.savez(KLs_save_path, **KLs_dict)
    print("\n KLs saved successfully.")

    print(f"\n Saving squared norms to {squared_norms_save_path}:")
    np.savez(squared_norms_save_path, **squared_norms_dict)
    print("\n Squared norms saved successfully.")

    if save_posteriors:
        print(f"\n Saving posteriors to {posteriors_dict_save_path}:")
        with open(posteriors_dict_save_path, "wb") as f:
            pickle.dump(posteriors_dict, f)
        print(f"\n Posteriors saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--x_observed", type=float, nargs="+", required=True)
    parser.add_argument("--num_sequential_rounds", type=int, default=4)
    parser.add_argument("--num_simulations_per_round", type=int, default=5000)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--L", type=float, default=-1.)
    parser.add_argument("--U", type=float, default=1.)
    parser.add_argument("--use_combined_loss", type=bool, default=False)
    parser.add_argument("--density_estimator", type=str, default="maf")
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--save_posteriors", type=bool, default=False)
    
    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds,
         args.num_simulations_per_round, args.d, args.L, args.U, args.use_combined_loss,
         args.density_estimator, args.num_repetitions, args.save_posteriors)