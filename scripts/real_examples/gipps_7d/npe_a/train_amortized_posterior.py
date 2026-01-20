import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.inference import NPE_A
import scipy
from examples.gipps import make_prior_7d_npe_a, simulator
import argparse
from pathlib import Path
import pickle
import yaml
import os

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "npe_a")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "trajectories")


def main(num_simulations, num_components,
        aL, aU,
        bL, bU,
        VL, VU,
        xf0L, xf0U,
        vf0L, vf0U,
        muL, muU,
        sigmasquaredL, sigmasquaredU,
        tau, N, ll, psi, bl, leader_trajectory_ID):
    
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
            "muL": muL,
            "muU": muU,
            "sigmasquaredL": sigmasquaredL,
            "sigmasquaredU": sigmasquaredU,
        }
    
    prior = make_prior_7d_npe_a(aL, aU,
                        bL, bU,
                        VL, VU,
                        xf0L, xf0U,
                        vf0L, vf0U,
                        muL, muU,
                        sigmasquaredL, sigmasquaredU)
    
    amortized_posterior_config = {
        "num_simulations": num_simulations,
        "num_components": num_components,
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
        "amortized_posterior_config": amortized_posterior_config,
        "prior_config": prior_config
    }
    
    # Retrieve xl, vl
    leader_trajectory = np.load(path_to_leader_trajectory)
    xl = leader_trajectory["xl"]
    vl = leader_trajectory["vl"]
    
    
    inference = NPE_A(prior=prior, num_components=num_components)
    print("Generating samples:")
    parameter_samples = prior.sample((num_simulations,))  # simulate parameters from prior
    data_samples = simulator(parameter_samples, tau, N, ll, psi, xl, vl, bl)  # simulate data for each parameter
    print("Samples generated successfully.")
    print("Training posterior:")
    inference = inference.append_simulations(parameter_samples, data_samples)
    density_estimator = inference.train(final_round=True)
    amortized_posterior = inference.build_posterior()
    print("Posterior trained successfully.")

    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_posterior{i}_leader_trajectory{leader_trajectory_ID}.pkl"):
        i += 1

    amortized_posterior_save_path = results_path + f"/amortized_posterior{i}_leader_trajectory{leader_trajectory_ID}" + ".pkl"
    config_save_path = results_path + f"/amortized_posterior{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"

    print(f"Saving trained posterior to {amortized_posterior_save_path}:")
    with open(amortized_posterior_save_path, "wb") as handle:
        pickle.dump(amortized_posterior, handle)
    print("Posterior saved successfully.")

    print(f"Saving config file to path {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("Config file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_simulations", type=int, default=20000)
    parser.add_argument("--num_components", type=int, default=1)

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
    parser.add_argument("--muL", type=float, default=-5.) # THIS DOESN'T COINCIDE WITH PAPER SINCE THIS WAS MADE UNIFORM WHEN IT SHOULD BE N(0,9)
    parser.add_argument("--muU", type=float, default=5.) # THIS DOESN'T COINCIDE WITH PAPER SINCE THIS WAS MADE UNIFORM WHEN IT SHOULD BE N(0,9)
    parser.add_argument("--sigmasquaredL", type=float, default=0.) # THIS DOESN'T COINCIDE WITH PAPER SINCE THIS WAS MADE UNIFORM WHEN IT SHOULD BE GAMMA(1,3)
    parser.add_argument("--sigmasquaredU", type=float, default=3.) # THIS DOESN'T COINCIDE WITH PAPER SINCE THIS WAS MADE UNIFORM WHEN IT SHOULD BE GAMMA(1,3)

    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--ll", type=float, default=7.5)
    parser.add_argument("--psi", type=float, default=1.05)
    parser.add_argument("--bl", type=float, default=-4.)
    parser.add_argument("--leader_trajectory_ID", type=int, required=True)

    args = parser.parse_args()
    main(args.num_simulations, args.num_components,
        args.aL, args.aU,
        args.bL, args.bU,
        args.VL, args.VU,
        args.xf0L, args.xf0U,
        args.vf0L, args.vf0U,
        args.muL, args.muU,
        args.sigmasquaredL, args.sigmasquaredU,
        args.tau, args.N, args.ll, args.psi, args.bl, args.leader_trajectory_ID)