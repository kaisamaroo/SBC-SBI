import numpy as np
import torch
import scipy
from examples.gipps import follower_trajectory_stochastic
import argparse
from pathlib import Path
import yaml
import os

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "npe_a")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "trajectories")


def main(af, bf, Vf, xf0, vf0, mu, sigmasquared, tau, N, psi, ll, bl, leader_trajectory_ID):
    leader_trajectory_name = f"leader_trajectory{leader_trajectory_ID}"
    path_to_leader_trajectory = trajectories_path + "/" + leader_trajectory_name + ".npz"
    path_to_leader_trajectory_config = trajectories_path + "/" + leader_trajectory_name + ".yaml"

    # Retrieve leader trajectory
    leader_trajectory = np.load(path_to_leader_trajectory)
    xl = leader_trajectory["xl"]
    vl = leader_trajectory["vl"]

    # Retrieve leader trajectory config
    with open(path_to_leader_trajectory_config, "r") as f:
        leader_trajectory_config = yaml.safe_load(f) # Dictionary

    # Follower trajectory config file
    follower_trajectory_config = {
                    "af": af,
                    "bf": bf,
                    "Vf": Vf,
                    "xf0": xf0,
                    "vf0": vf0,
                    "mu": mu,
                    "sigmasquared": sigmasquared,
                    "tau": tau,
                    "N": N,
                    "psi": psi,
                    "ll": ll,
                    "bl": bl,
                    "leader_trajectory_ID": leader_trajectory_ID,
                }
    # Cumulative config file
    config = {"leader_trajectory_config": leader_trajectory_config,
              "follower_trajectory_config": follower_trajectory_config}

    # Generate the follower trajectory with stochastic noise
    print("Generating xf, vf:")
    xf, vf = follower_trajectory_stochastic(af, bf, Vf, xf0, vf0, mu, sigmasquared, tau, N, ll, psi, xl, vl, bl)
    print("xf, vf generated successfully.")

    # Find next ID
    i = 0
    while os.path.exists(trajectories_path + f"/follower_trajectory{i}_leader_trajectory{leader_trajectory_ID}.npz"):
        i += 1

    follower_trajectory_save_path = trajectories_path + f"/follower_trajectory{i}_leader_trajectory{leader_trajectory_ID}" + ".npz"
    config_save_path = trajectories_path + f"/follower_trajectory{i}_leader_trajectory{leader_trajectory_ID}" + ".yaml"

    print(f"Saving xf, vf to path {follower_trajectory_save_path}:")
    np.savez(follower_trajectory_save_path, xf=xf, vf=vf)
    print("xf, vf saved succesfully.")

    print(f"Saving config file to path {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("Config file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--af", type=float, default=1.797)
    parser.add_argument("--bf", type=float, default=-3.566)
    parser.add_argument("--Vf", type=float, default=28.)
    parser.add_argument("--xf0", type=float, default=-22.521)
    parser.add_argument("--vf0", type=float, default=10.148)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--sigmasquared", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--ll", type=float, default=7.5)
    parser.add_argument("--psi", type=float, default=1.05)
    parser.add_argument("--bl", type=float, default=-4.)
    parser.add_argument("--leader_trajectory_ID", type=int, required=True)
    args = parser.parse_args()
    main(args.af, args.bf, args.Vf, args.xf0, args.vf0, args.mu, args.sigmasquared, args.tau, args.N, args.psi, args.ll, args.bl, args.leader_trajectory_ID)
