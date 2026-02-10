import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign # Kolmogorov distribution
import torch
from sbi.inference import NPE_A, NPE_C
from sbi.utils import RestrictedPrior, get_density_thresholder


def plot_sbc_histogram(ranks, N_samp, title=None, ax=None):
    ranks = np.array(ranks)

    N_iter = np.sum(1 - np.isnan(ranks))

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    # Plot Histogram
    ax.hist(ranks, density=True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"SBC histogram " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()

    return ax


def plot_sbc_ecdf(ranks, N_samp, alpha=0.05, title=None, ax=None):
    ranks = np.array(ranks)

    N_iter = np.sum(1 - np.isnan(ranks))

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    unit_range = np.linspace(0, 1, 1000)
    x_grid = np.arange(0, N_samp + 1) / N_samp

    k_alpha = kstwobign.ppf(1-alpha) # (1-alpha) quantile of Kolmogorov distribution

    # Plot approximate asymptotic 1-alpha simultaneous credible region for ECDF
    ax.fill_between(unit_range,
        unit_range - k_alpha / np.sqrt(N_iter),
        unit_range + k_alpha / np.sqrt(N_iter),
        alpha=0.3)
    
    ECDF = np.vectorize(lambda x : np.sum(ranks <= x) / N_iter)

    # Plot ECDF
    ax.plot(x_grid, ECDF(x_grid))

    ax.set_xlabel("x")
    ax.set_ylabel("ECDF")
    ax.set_ylim((0,1))
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Sample ECDF with approximate {100*(1-alpha)}% interval " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()

    return ax


def plot_sbc_ecdf_diff(ranks, N_samp, alpha=0.05, title=None, ax=None):
    ranks = np.array(ranks)

    N_iter = np.sum(1 - np.isnan(ranks))

    internally_defined_ax = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        internally_defined_ax = True

    unit_range = np.linspace(0, 1, 1000)
    x_grid = np.arange(0, N_samp + 1) / N_samp

    k_alpha = kstwobign.ppf(1-alpha) # (1-alpha) quantile of Kolmogorov distribution

    # Plot approximate asymptotic 1-alpha simultaneous credible region for ECDF
    ax.fill_between(unit_range,
        -1 * k_alpha / np.sqrt(N_iter),
        k_alpha / np.sqrt(N_iter),
        alpha=0.3)
    
    ECDF = np.vectorize(lambda x : np.sum(ranks <= x) / N_iter)

    # Plot ECDF
    ax.plot(x_grid, ECDF(x_grid) - x_grid)

    ax.set_xlabel("x")
    ax.set_ylabel(r"ECDF($x$) - $x$")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(r"ECDF($x$) - $x$ " + f"with approximate {100*(1-alpha)}% interval " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")
    
    if internally_defined_ax:
        plt.show()
    return ax


def plot_sbc_all(ranks, N_samp, alpha=0.05, title=None):
    ranks = np.array(ranks)

    N_iter = np.sum(1 - np.isnan(ranks))

    fig, ax = plt.subplots(figsize=(10, 5), ncols=3)
    plot_sbc_histogram(ranks, N_samp=N_samp, title="Histogram", ax=ax[0])
    plot_sbc_ecdf(ranks, N_samp=N_samp, alpha=alpha, title=f"ECDF with {100*(1-alpha)}% interval", ax=ax[1])
    plot_sbc_ecdf_diff(ranks, N_samp=N_samp, alpha=alpha, title=f"ECDF - x with {100*(1-alpha)}% interval", ax=ax[2])
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle("SBC diagnostic plots " + r"($N_{samp} =$ " + f"{N_samp}" + r" and $N_{iter} =$ " + f"{N_iter})")

    plt.tight_layout()
    plt.show()
    return ax
    

def sbc_ranks(model, prior, posterior, test_function=None, N_iter=100, N_samp=100, show_progress=False, return_samples=False, always_return_dict=False):
    """
    return normalized SBC ranks
    """
    if return_samples:
        samples_dict = {}

    if isinstance(test_function, list):
        # Multiple test functions given
        # test_function should be of the form [(test_function, test_function_name), ...]

        # Ranks will be of form {"test_function_name": test_function_ranks, ...} (to be saved as .npz)
        ranks = {_[1]: [] for _ in test_function}
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = model(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_samples = posterior.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = np.array(posterior_samples)
            # Calculate rank for each test function
            for test_function_, test_function_name_ in test_function:
                ranks[test_function_name_].append(
                    float(torch.sum(test_function_(prior_sample) * torch.ones(N_samp) > test_function_(posterior_samples))/N_samp)
                    )
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        # Convert rank lists to np arrays
        ranks = {k: np.array(v) for k, v in ranks.items()}
        if return_samples:
            return ranks, samples_dict
        else:
            return ranks

    else:
        # Single (or no) test function given
        ranks = []
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = model(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_samples = posterior.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = np.array(posterior_samples)
            if test_function:
                rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
            else:
                # If no test function provided, assume theta is 1D.
                rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
            ranks.append(float(rank))
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        if return_samples:
            if always_return_dict:
                return {"": np.array(ranks)}, samples_dict
            else:
                return np.array(ranks), samples_dict
        else:
            if always_return_dict:
                return {"": np.array(ranks)}
            else:
                return np.array(ranks)


# OLD FUNCTION, ONLY USED FOR OLD SCRIPTS. COULD CHANGE ALL OLD SCRIPTS AND DELETE THIS.
def sbc_ranks_and_samples(model, prior, posterior, test_function=None, N_iter=100, N_samp=100, show_progress=False):
    """
    OLD VERSION. MODERN APPLICATIONS SHOULD USE sbc_ranks WITH return_samples SET TO True

    return normalized SBC ranks AND a sictionary containing each round's samples
    """
    # YET TO IMPLEMENT MULTIPLE TEST FUNCTIONS
    ranks = []
    samples_dict = {}
    print_indices = [(i * N_iter) // 10 for i in range(10)]
    for i in range(N_iter):
        if show_progress and i in print_indices:
            print(f"SBC round {i} out of {N_iter} ({100 * i / N_iter}%)")
        prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
        simulated_datapoint = model(prior_sample) # Simulate a datapoint from the model given the prior sample. Returns 1d tensor
        posterior_samples = posterior.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.

        samples_dict[f"prior_sample_round_{i}"] = prior_sample
        samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
        samples_dict[f"posterior_samples_round_{i}"] = np.array(posterior_samples)

        if test_function:
            rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
        else:
            # If no test function provided, assume theta is 1D.
            rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
        ranks.append(float(rank))
    if show_progress:
        print("SBC complete")
    return ranks, samples_dict


def train_snpe_a_posterior(simulator, prior, x_observed, num_sequential_rounds, num_simulations_per_round, num_components):
    inference = NPE_A(prior, num_components=num_components)
    proposal = prior
    for r in range(num_sequential_rounds):
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)
        # SNPE-A trains a Gaussian density estimator in all but the last round. In the last round,
        # it trains a mixture of Gaussians, which is why we have to pass the `final_round` flag.
        if r == num_sequential_rounds - 1:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(final_round=True)
            sequential_posterior = inference.build_posterior() # Don't set default x for returned posterior
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(final_round=False)
            sequential_posterior = inference.build_posterior().set_default_x(x_observed)
            proposal = sequential_posterior
    return sequential_posterior


def sbc_ranks_snpe_a(simulator,
                         prior,
                         train_sequential_posterior,
                         test_function=None,
                         N_iter=100,
                         N_samp=100,
                         num_sequential_rounds=4,
                         num_simulations_per_round=5000,
                         num_components=1,
                         show_progress=False):
    """
    return normalized SBC ranks for SNPE-A, being careful to account for errors
    due to non-spd covariance matrices
    """
    failed_round_counter = 0
    if isinstance(test_function, list):
        # Multiple test functions given
        # test_function should be of the form [(test_function, test_function_name), ...]

        # Ranks will be of form {"test_function_name": test_function_ranks, ...} (to be saved as .npz)
        ranks = {_[1]: [] for _ in test_function}
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            try:
                posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, num_components)
                try:
                    posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
                    # Calculate rank for each test function
                    for test_function_, test_function_name_ in test_function:
                        ranks[test_function_name_].append(
                            float(torch.sum(test_function_(prior_sample) * torch.ones(N_samp) > test_function_(posterior_samples))/N_samp)
                            )
                except AssertionError:
                    print(f"\n SBC ROUND {i+1} FINAL POSTERIOR NOT SPD! SKIPPING ROUND")
                    ranks.append(np.nan)
                    failed_round_counter += 1
            except AssertionError:
                print(f"\n SBC ROUND {i+1} HAD A PROPOSAL PRIOR NOT SPD! SKIPPING ROUND")
                ranks.append(np.nan)
                failed_round_counter += 1
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC WITH {failed_round_counter} FAILED ROUNDS" + 12*"-")
        ranks = {k: np.array(v) for k, v in ranks.items()}
        return ranks

    else:
        # Single (or no) test function given
        ranks = []
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            try:
                posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, num_components)
                try:
                    posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
                    if test_function:
                        rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
                    else:
                        # If no test function provided, assume theta is 1D.
                        rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
                    ranks.append(float(rank))
                except AssertionError:
                    print(f"\n SBC ROUND {i+1} FINAL POSTERIOR NOT SPD! SKIPPING ROUND")
                    ranks.append(np.nan)
                    failed_round_counter += 1
            except AssertionError:
                print(f"\n SBC ROUND {i+1} HAD A PROPOSAL PRIOR NOT SPD! SKIPPING ROUND")
                ranks.append(np.nan)
                failed_round_counter += 1
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC WITH {failed_round_counter} FAILED RUNS" + 12*"-")
        return np.array(ranks)


def sbc_ranks_snpe_a_and_samples(simulator,
                         prior,
                         train_sequential_posterior,
                         test_function=None,
                         N_iter=100,
                         N_samp=100,
                         num_sequential_rounds=4,
                         num_simulations_per_round=5000,
                         num_components=1,
                         show_progress=False):
    """
    return normalized SBC ranks for SNPE-A, being careful to account for errors
    due to non-spd covariance matrices
    """
    if isinstance(test_function, list):
        raise NotImplementedError("Multiple test functions yet to be implenented.")
    failed_round_counter = 0
    ranks = []
    samples_dict = {}
    for i in range(N_iter):
        if show_progress:
            print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
        prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
        simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
        samples_dict[f"prior_sample_round_{i}"] = prior_sample
        samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
        try:
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, num_components)
            try:
                posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
                samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
                if test_function:
                    rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
                else:
                    # If no test function provided, assume theta is 1D.
                    rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
                ranks.append(float(rank))
            except AssertionError:
                print(f"\n SBC ROUND {i+1} FINAL POSTERIOR NOT SPD! SKIPPING ROUND")
                ranks.append(np.nan)
                samples_dict[f"posterior_samples_round_{i}"] = np.nan
                failed_round_counter += 1
        except AssertionError:
            print(f"\n SBC ROUND {i+1} HAD A PROPOSAL PRIOR NOT SPD! SKIPPING ROUND")
            ranks.append(np.nan)
            samples_dict[f"posterior_samples_round_{i}"] = np.nan
            failed_round_counter += 1
    if show_progress:
        print("\n" + 12*"-" + f"FINISHED SBC WITH {failed_round_counter} FAILED RUNS" + 12*"-")
    return ranks, samples_dict


def train_snpe_c_posterior(simulator, prior, x_obs, num_sequential_rounds, num_simulations_per_round, use_combined_loss=False, show_progress=True, sample_with='direct'):
    """
    Return x_obs-sequentially-trained SNPE-C posterior 
    """
    inference = NPE_C(prior)
    proposal = prior
    if use_combined_loss:
        print("\n Using combined loss.")
    for r in range(num_sequential_rounds):
        if show_progress:
            print(f"\n ------------- SEQUENTIAL ROUND {r+1} out of {num_sequential_rounds} -------------")
            print(f"Generating {num_simulations_per_round} training samples")
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)
        if show_progress:
            print("Training samples generated")

        if r == num_sequential_rounds - 1:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            # Final posterior shouldnt be conditioned on x_obs (so we can see how it infers other x)
            sequential_posterior = inference.build_posterior(sample_with=sample_with)
        else:
            _ = inference.append_simulations(parameter_samples, data_samples, proposal=proposal).train(use_combined_loss=use_combined_loss)
            sequential_posterior = inference.build_posterior(sample_with=sample_with).set_default_x(x_obs)
            proposal = sequential_posterior
    return sequential_posterior


def sbc_ranks_snpe_c(simulator,
                    prior,
                    train_sequential_posterior,
                    test_function=None,
                    N_iter=100,
                    N_samp=100,
                    num_sequential_rounds=4,
                    num_simulations_per_round=5000,
                    use_combined_loss=False,
                    show_progress=True,
                    return_samples=False,
                    always_return_dict=False,
                    sample_with="direct"):
    """
    return normalized SBC ranks for SNPE-C.
    """
    if return_samples:
        samples_dict = {}

    if isinstance(test_function, list):
        # Multiple test functions given
        # test_function should be of the form [(test_function, test_function_name), ...]

        # Ranks will be of form {"test_function_name": test_function_ranks, ...} (to be saved as .npz)
        ranks = {_[1]: [] for _ in test_function}
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, use_combined_loss=use_combined_loss, sample_with=sample_with)
            posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
            # Calculate rank for each test function
            for test_function_, test_function_name_ in test_function:
                ranks[test_function_name_].append(
                    float(torch.sum(test_function_(prior_sample) * torch.ones(N_samp) > test_function_(posterior_samples))/N_samp)
                    )
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        # Convert rank lists to np arrays
        ranks = {k: np.array(v) for k, v in ranks.items()}
        if return_samples:
            return ranks, samples_dict
        else:
            return ranks

    else:
        # Single (or no) test function given
        ranks = []
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, use_combined_loss=use_combined_loss, sample_with=sample_with)
            posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
            if test_function:
                rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
            else:
                # If no test function provided, assume theta is 1D.
                rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
            ranks.append(float(rank))
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        if return_samples:
            if always_return_dict:
                return {"": np.array(ranks)}, samples_dict
            else:
                return np.array(ranks), samples_dict
        else:
            if always_return_dict:
                return {"": np.array(ranks)}
            else:
                return np.array(ranks)
            

def train_tsnpe_posterior(simulator, prior, x_obs, num_sequential_rounds, num_simulations_per_round, 
                          show_progress=True, sample_with='direct', restricted_prior_sample_with="direct", epsilon=1e-4,
                          num_samples_to_estimate_support=1000000):
    """
    Return x_obs-sequentially-trained TSNPE posterior 
    """
    inference = NPE_C(prior)
    proposal = prior

    for r in range(num_sequential_rounds):
        if show_progress:
            print(f"\n ------------- SEQUENTIAL ROUND {r} out of {num_sequential_rounds} -------------")
            print(f"Generating {num_simulations_per_round} training samples")
        parameter_samples = proposal.sample((num_simulations_per_round,))
        data_samples = simulator(parameter_samples)
        if show_progress:
            print("Training samples generated")

        if r == num_sequential_rounds - 1:
            _ = inference.append_simulations(parameter_samples, data_samples).train(force_first_round_loss=True)
            # Final posterior shouldnt be conditioned on x_obs (so we can see how it infers other x)
            sequential_posterior = inference.build_posterior(sample_with=sample_with)
        else:
            if show_progress:
                print("\n Training density estimator")
            _ = inference.append_simulations(parameter_samples, data_samples).train(force_first_round_loss=True)
            if show_progress:
                print("\n Density estimator trained")
            sequential_posterior = inference.build_posterior(sample_with=sample_with).set_default_x(x_obs)
            if show_progress:  
                print(f"\n Approximating HPR region of density estimator with {num_samples_to_estimate_support} samples")
            accept_reject_fn = get_density_thresholder(sequential_posterior, quantile=epsilon, num_samples_to_estimate_support=num_samples_to_estimate_support)
            if show_progress:
                print("\n HPR region of density estimator approximated successfully.")
            proposal = RestrictedPrior(prior, accept_reject_fn, sample_with=restricted_prior_sample_with)
    return sequential_posterior


def sbc_ranks_tsnpe(simulator,
                    prior,
                    train_sequential_posterior,
                    test_function=None,
                    N_iter=100,
                    N_samp=100,
                    num_sequential_rounds=4,
                    num_simulations_per_round=5000,
                    show_progress=True,
                    return_samples=False,
                    always_return_dict=False,
                    sample_with="direct",
                    restricted_prior_sample_with="direct",
                    epsilon=1e-4,
                    num_samples_to_estimate_support=1000000):
    """
    return normalized SBC ranks for TSNPE.
    """
    if return_samples:
        samples_dict = {}

    if isinstance(test_function, list):
        # Multiple test functions given
        # test_function should be of the form [(test_function, test_function_name), ...]

        # Ranks will be of form {"test_function_name": test_function_ranks, ...} (to be saved as .npz)
        ranks = {_[1]: [] for _ in test_function}
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, 
                          show_progress=show_progress, sample_with=sample_with, restricted_prior_sample_with=restricted_prior_sample_with, epsilon=epsilon,
                          num_samples_to_estimate_support=num_samples_to_estimate_support)
            posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
            # Calculate rank for each test function
            for test_function_, test_function_name_ in test_function:
                ranks[test_function_name_].append(
                    float(torch.sum(test_function_(prior_sample) * torch.ones(N_samp) > test_function_(posterior_samples))/N_samp)
                    )
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        # Convert rank lists to np arrays
        ranks = {k: np.array(v) for k, v in ranks.items()}
        if return_samples:
            return ranks, samples_dict
        else:
            return ranks

    else:
        # Single (or no) test function given
        ranks = []
        for i in range(N_iter):
            if show_progress:
                print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
            prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
            simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
            posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round, 
                          show_progress=show_progress, sample_with=sample_with, restricted_prior_sample_with=restricted_prior_sample_with, epsilon=epsilon,
                          num_samples_to_estimate_support=num_samples_to_estimate_support)
            posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
            if return_samples:
                samples_dict[f"prior_sample_round_{i}"] = prior_sample
                samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
                samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
            if test_function:
                rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
            else:
                # If no test function provided, assume theta is 1D.
                rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
            ranks.append(float(rank))
        if show_progress:
            print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
        if return_samples:
            if always_return_dict:
                return {"": np.array(ranks)}, samples_dict
            else:
                return np.array(ranks), samples_dict
        else:
            if always_return_dict:
                return {"": np.array(ranks)}
            else:
                return np.array(ranks)


# OLD FUNCTION, ONLY USED FOR OLD SCRIPTS. COULD CHANGE ALL OLD SCRIPTS AND DELETE THIS.
def sbc_ranks_snpe_c_and_samples(simulator,
                    prior,
                    train_sequential_posterior,
                    test_function=None,
                    N_iter=100,
                    N_samp=100,
                    num_sequential_rounds=4,
                    num_simulations_per_round=5000,
                    show_progress=True):
    """
    OLD VERSION. MODERN APPLICATIONS SHOULD USE sbc_ranks WITH return_samples SET TO True

    return normalized SBC ranks for SNPE-C
    """
    if isinstance(test_function, list):
        raise NotImplementedError("Multiple test functions yet to be implenented.")
    ranks = []
    samples_dict = {}
    for i in range(N_iter):
        if show_progress:
            print("\n" + 12*"-" + f"SBC ROUND {i+1} OUT OF {N_iter}" + 12*"-")
        prior_sample = prior.sample() # Sample from prior. Returns 1D tensor
        simulated_datapoint = simulator(prior_sample) # Simulate a datapoint from the simulator given the prior sample. Returns 1d tensor
        posterior_sequential = train_sequential_posterior(simulator, prior, simulated_datapoint, num_sequential_rounds, num_simulations_per_round)
        posterior_samples = posterior_sequential.sample((N_samp,), x=simulated_datapoint, show_progress_bars=False) # Numpy array of (num_samples, ) samples.
        samples_dict[f"prior_sample_round_{i}"] = prior_sample
        samples_dict[f"data_sample_round_{i}"] = simulated_datapoint
        samples_dict[f"posterior_samples_round_{i}"] = posterior_samples
        if test_function:
            rank = torch.sum(test_function(prior_sample) * torch.ones(N_samp) > test_function(posterior_samples))/N_samp
        else:
            # If no test function provided, assume theta is 1D.
            rank = torch.sum(prior_sample.item() * torch.ones_like(posterior_samples) > posterior_samples)/N_samp
        ranks.append(float(rank))
    if show_progress:
        print("\n" + 12*"-" + f"FINISHED SBC" + 12*"-")
    return ranks, samples_dict