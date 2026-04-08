import torch
from sbi.inference import NPE
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
from sbi.utils import RestrictedPrior, get_density_thresholder
import argparse
from pathlib import Path
import os
import yaml
import time
import math

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "tsnpe")


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
         d, L, U, density_estimator, num_repetitions, epsilon, restricted_prior_sample_with, 
         num_samples_to_estimate_support, save_posteriors):
    
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
        inference = NPE(prior=prior, density_estimator=density_estimator)
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

            if r > 0:
                # Note that proposal is not the posterior approximation, it is sequential_posterior.
                # thus, we need to generate samples from the posterior approximation to get the squared_norms and approximate the KL etc.
                q_samples = sequential_posterior.sample((num_simulations_per_round,))                

                print("\n Computing squared norms:")
                # Save squared norms of parameter samples 
                squared_norms = (q_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
                squared_norms_dict[f"round_{r - 1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
                print("\n Squared norms generated")

                print("\n Computing KLs:")
                # Save approximate (reverse) KL divergence:
                # Compute log density of sequential_posterior
                log_q = sequential_posterior.log_prob(q_samples) # Tensor of shape (num_simulations_per_round, ). Note set_default_x has already been called
                # Compute log density of true posterior
                log_p = true_posterior_log_p(q_samples, torch.tensor(x_observed), sigma, d) # Tensor of shape (num_simulations_per_round, )
                KL = torch.mean(log_q - log_p).item() # Monte carlo approximation to reverse KL
                KLs_dict[f"round_{r - 1}"][rep] = KL
                print("\n KLs generated")

            print("\n Training proposal:")
            training_start_time = time.perf_counter()
            if r == num_sequential_rounds - 1:
                density_estimator_ = inference.append_simulations(parameter_samples, data_samples).train(force_first_round_loss=True)
                sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
                if save_posteriors:
                    posteriors_dict[f"repetition_{rep}"][f"round_{r}"] = sequential_posterior
            else:
                density_estimator_ = inference.append_simulations(parameter_samples, data_samples).train(force_first_round_loss=True)
                sequential_posterior = inference.build_posterior().set_default_x(torch.tensor(x_observed))
                if save_posteriors:
                    posteriors_dict[f"repetition_{rep}"][f"round_{r}"] = sequential_posterior
                accept_reject_fn = get_density_thresholder(sequential_posterior, quantile=epsilon, num_samples_to_estimate_support=num_samples_to_estimate_support)
                proposal = RestrictedPrior(prior, accept_reject_fn, sample_with=restricted_prior_sample_with, posterior=sequential_posterior)
            training_end_time = time.perf_counter()
            training_time = training_end_time - training_start_time
            avg_training_times[f"round_{r}"] += training_time / num_repetitions
            print("\n Proposal trained successfully:")
        print("\n Posterior trained successfully.")

        # Note that proposal is not the posterior approximation, it is sequential_posterior.
        # thus, we need to generate samples from the posterior approximation to get the squared_norms and approximate the KL etc.
        q_samples = sequential_posterior.sample((num_simulations_per_round,), x=torch.tensor(x_observed))                

        print("\n Computing squared norms:")
        # Save squared norms of parameter samples 
        squared_norms = (q_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
        squared_norms_dict[f"round_{num_sequential_rounds - 1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
        print("\n Squared norms generated")

        print("\n Computing KLs:")
        # Save approximate (reverse) KL divergence:
        # Compute log density of density estimator
        log_q = sequential_posterior.log_prob(q_samples, x=torch.tensor(x_observed)) # Tensor of shape (num_simulations_per_round, ). Note set_default_x has already been called
        # Compute log density of true posterior
        log_p = true_posterior_log_p(q_samples, torch.tensor(x_observed), sigma, d) # Tensor of shape (num_simulations_per_round, )
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
              "density_estimator": density_estimator,
              "epsilon": epsilon,
              "restricted_prior_sample_with": restricted_prior_sample_with,
              "num_samples_to_estimate_support": num_samples_to_estimate_support,
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
    parser.add_argument("--density_estimator", type=str, default="maf")
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--restricted_prior_sample_with", type=str, default="rejection")
    parser.add_argument("--num_samples_to_estimate_support", type=int, default=1000000)
    parser.add_argument("--save_posteriors", type=bool, default=False)

    
    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds,
         args.num_simulations_per_round, args.d, args.L, args.U,
         args.density_estimator, args.num_repetitions,
         args.epsilon, args.restricted_prior_sample_with, args.num_samples_to_estimate_support,
         args.save_posteriors)