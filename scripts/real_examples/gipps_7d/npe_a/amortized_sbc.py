import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
from sbi.inference import NPE_A
from sbi.analysis import pairplot
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from torch.distributions import Exponential, Normal, InverseGamma, MultivariateNormal
from sbi.utils import BoxUniform
from examples.gipps import make_prior_7d_npe_a, simulator, get_test_function
from sbc.sbc_tools import sbc_ranks
import argparse
from pathlib import Path
import pickle
import yaml
import os

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "npe_a")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "trajectories")


def main(N_iter, N_samp, amortized_posterior_ID, leader_trajectory_ID, test_function_name,
        aL, aU,
        bL, bU,
        VL, VU,
        xf0L, xf0U,
        vf0L, vf0U,
        muL, muU,
        sigmasquaredL, sigmasquaredU,
        tau, N, ll, psi, bl):
    
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

    sbc_config = {
        "N_iter": N_iter,
        "N_samp": N_samp,
        "test_function_name": test_function_name,
        "tau": tau, 
        "N": N,
        "ll": ll, 
        "psi": psi,
        "bl": bl, 
        "leader_trajectory_ID": leader_trajectory_ID,
        "amortized_posterior_ID": amortized_posterior_ID
    }

    leader_trajectory_name = f"leader_trajectory{leader_trajectory_ID}"
    path_to_leader_trajectory = trajectories_path + "/" + leader_trajectory_name + ".npz"
    path_to_leader_trajectory_config = trajectories_path + "/" + leader_trajectory_name + ".yaml"

    amortized_posterior_name = f"amortized_posterior{amortized_posterior_ID}_leader_trajectory{leader_trajectory_ID}"
    path_to_amortized_posterior = results_path + "/" + amortized_posterior_name + ".pkl"
    path_to_amortized_posterior_config = results_path + "/" + amortized_posterior_name + ".yaml"

    # Retrieve leader trajectory config
    with open(path_to_leader_trajectory_config, "r") as f:
        leader_trajectory_config = yaml.safe_load(f) # Dictionary

    # Retrieve amortized posterior config
    with open(path_to_amortized_posterior_config, "r") as f:
        amortized_posterior_config = yaml.safe_load(f) # Dictionary

    config = {
        "leader_trajectory_config": leader_trajectory_config,
        "sbc_config": sbc_config,
        "amortized_posterior_config": amortized_posterior_config,
        "prior_config": prior_config
    }
    
    # Retrieve xl, vl
    leader_trajectory = np.load(path_to_leader_trajectory)
    xl = leader_trajectory["xl"]
    vl = leader_trajectory["vl"]

    # Retrieve amortized posterior
    with open(path_to_amortized_posterior, "rb") as f:
        amortized_posterior = pickle.load(f)
    
    # Retrieve test function
    test_function = get_test_function(test_function_name)
    # Ensure simulator is in correct format
    simulator_ = lambda x: simulator(x, tau, N, ll, psi, xl, vl, bl)
    # Generate ranks
    print("Generating ranks:")
    ranks = sbc_ranks(simulator_, prior, amortized_posterior, test_function=test_function, N_iter=N_iter, N_samp=N_samp, show_progress=True)
    print("Ranks generated successfully.")

    # Find next ID
    i = 0
    while os.path.exists(results_path + f"/amortized_sbc{i}_amortized_posterior{amortized_posterior_ID}_leader_trajectory{leader_trajectory_ID}.npy"):
        i += 1

    amortized_sbc_save_path = results_path + f"/amortized_sbc{i}_amortized_posterior{amortized_posterior_ID}_leader_trajectory{leader_trajectory_ID}" + ".npy"
    config_save_path = results_path + f"/amortized_sbc{i}_amortized_posterior{amortized_posterior_ID}_leader_trajectory{leader_trajectory_ID}" + ".yaml"

    print(f"Saving ranks to {amortized_sbc_save_path}:")
    np.save(amortized_sbc_save_path, ranks)
    print("Ranks saved successfully.")

    print(f"Saving config file to path {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("Config file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--N_samp", type=int, default=100)
    parser.add_argument("--test_function_name", type=str, default="projection0")

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
    parser.add_argument("--amortized_posterior_ID", type=int, required=True)

    args = parser.parse_args()
    main(args.N_iter, args.N_samp, args.amortized_posterior_ID, args.leader_trajectory_ID, args.test_function_name,
        args.aL, args.aU,
        args.bL, args.bU,
        args.VL, args.VU,
        args.xf0L, args.xf0U,
        args.vf0L, args.vf0U,
        args.muL, args.muU,
        args.sigmasquaredL, args.sigmasquaredU,
        args.tau, args.N, args.ll, args.psi, args.bl)
