import torch
from sbi.inference import NLE
import numpy as np
import pickle
from examples.unif_norm import make_prior, simulator
import argparse
from pathlib import Path
import os
import yaml
import time

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "toy_examples" / "unif_norm" / "nle_mcmc")


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


def gaussian_log_density(theta, mu, Sigma, d):
    """
    Computes log density of a multivariate normal N(mu, Sigma) evaluated at each row of theta.
    mu is a torch.Tensor with shape (d,), mean of the MVN.
    Sigma is a torch.Tensor with shape (d, d), covariance matrix (must be positive definite).
    returns log_prob, a torch.Tensor of shape (batch_size,), log-density of each row of theta.
    """
    
    # Compute difference
    diff = theta - mu  # (batch_size, d)

    # Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(Sigma)  # (d, d)
    # Solve L y = diff^T for y (equivalent to Sigma^{-1} diff)
    # Use triangular solve
    solve = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (d, batch_size)

    # Squared Mahalanobis distance
    sq_mahalanobis = (solve ** 2).sum(dim=0)  # (batch_size,)

    # Log determinant using Cholesky
    log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))  # log|Sigma|

    log_norm_const = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi)) - 0.5 * log_det

    log_prob = log_norm_const - 0.5 * sq_mahalanobis  # (batch_size,)

    return log_prob


def gaussians_KL(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    Computes KL(q || p) where
        q = N(mu_q, Sigma_q)
        p = N(mu_p, Sigma_p)
    Args:
        mu_q: (d,) tensor
        Sigma_q: (d,d) positive definite tensor
        mu_p: (d,) tensor
        Sigma_p: (d,d) positive definite tensor
    Returns:
        scalar tensor (KL divergence)
    """

    d = mu_q.shape[0]

    # Cholesky decompositions
    L_q = torch.linalg.cholesky(Sigma_q)
    L_p = torch.linalg.cholesky(Sigma_p)

    # log determinants via Cholesky
    logdet_q = 2.0 * torch.sum(torch.log(torch.diag(L_q)))
    logdet_p = 2.0 * torch.sum(torch.log(torch.diag(L_p)))

    # Compute Sigma_p^{-1} via Cholesky solve
    Sigma_p_inv = torch.cholesky_inverse(L_p)

    # Trace term
    trace_term = torch.trace(Sigma_p_inv @ Sigma_q)

    # Quadratic term
    diff = (mu_p - mu_q).unsqueeze(1)  # (d,1)
    quad_term = (diff.T @ Sigma_p_inv @ diff).squeeze()

    kl = 0.5 * (
        logdet_p - logdet_q
        - d
        + trace_term
        + quad_term
    )

    return kl


def sample_covariance_matrix(theta, mu_hat):
    N = theta.shape[0] # number of samples
    diff = theta - mu_hat # (N, d)
    # Compute MLE covariance
    Sigma = (diff.T @ diff) / (N-1)  # (d, d)
    return Sigma


def main(sigma, x_observed, num_sequential_rounds, num_simulations_per_round,
         d, L, U, density_estimator, num_repetitions, mcmc_method, save_posteriors):
    
    if save_posteriors in ["True", "true"]:
        save_posteriors = True
        posteriors_dict = {f"repetition_{r}": {} for r in range(num_repetitions)}
    else:
        save_posteriors = False

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
        inference = NLE(prior=prior, density_estimator=density_estimator)
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
                # Save approximate (reverse) KL divergence by fitting a gaussian to samples:
                # Fit Gaussian parameters to the training data
                mu_q = parameter_samples.mean(axis=0) # Sample mean (MLE for a Gaussian fit)
                Sigma_q = sample_covariance_matrix(parameter_samples, mu_q)
                # Define mean and cov of true Gaussian posterior (assuming L is large so truncated \approx untruncated)
                mu_p = torch.tensor(x_observed)
                Sigma_p = sigma ** 2 * torch.eye(d)
                KL = gaussians_KL(mu_q, Sigma_q, mu_p, Sigma_p)
                KLs_dict[f"round_{r-1}"][rep] = KL
                print("\n KLs generated")


            print("\n Training proposal:")
            training_start_time = time.perf_counter()
            ########## TRAINING (no need for round split)
            density_estimator_ = inference.append_simulations(parameter_samples, data_samples).train()
            sequential_posterior = inference.build_posterior(sample_with="mcmc", mcmc_method=mcmc_method).set_default_x(torch.tensor(x_observed))
            if save_posteriors:
                posteriors_dict[f"repetition_{rep}"][f"round_{r}"] = sequential_posterior
            proposal = sequential_posterior
            ##########
            training_end_time = time.perf_counter()
            training_time = training_end_time - training_start_time
            avg_training_times[f"round_{r}"] += training_time / num_repetitions
            print("\n Proposal trained successfully:")
        print("\n Posterior trained successfully.")

        # Append final posterior squared norm and KL
        parameter_samples = sequential_posterior.sample((num_simulations_per_round,))

        print("\n Computing final round squared norms:")
        # Save squared norms of parameter samples 
        squared_norms = (parameter_samples**2).sum(axis=1) # Shape (num_simulations_per_round, )
        squared_norms_dict[f"round_{num_sequential_rounds-1}"][rep, :] = squared_norms # Shape (num_simulations_per_round, )
        print("\n Squared norms generated")

        print("\n Computing final round KLs:")
        # Save approximate (reverse) KL divergence by fitting a gaussian to samples:
        # Fit Gaussian parameters to the training data
        mu_q = parameter_samples.mean(axis=0) # Sample mean (MLE for a Gaussian fit)
        Sigma_q = sample_covariance_matrix(parameter_samples, mu_q)
        # Define mean and cov of true Gaussian posterior (assuming L is large so truncated \approx untruncated)
        mu_p = torch.tensor(x_observed)
        Sigma_p = sigma ** 2 * torch.eye(d)
        KL = gaussians_KL(mu_q, Sigma_q, mu_p, Sigma_p)
        KLs_dict[f"round_{num_sequential_rounds-1}"][rep] = KL
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
              "num_repetitions": num_repetitions,
              "mcmc_method": mcmc_method,
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
    parser.add_argument("--mcmc_method", type=str, default="slice_np_vectorized")
    parser.add_argument("--save_posteriors", type=str, default="False")


    args = parser.parse_args()
    main(args.sigma, args.x_observed, args.num_sequential_rounds,
         args.num_simulations_per_round, args.d, args.L, args.U,
         args.density_estimator, args.num_repetitions, args.mcmc_method, 
         args.save_posteriors)