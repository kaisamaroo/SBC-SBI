import numpy as np
import torch
import scipy
from examples.gipps import simulate_leader_trajectory
import argparse
from pathlib import Path
import os
import yaml

path_to_repo = Path(__file__).resolve().parents[4]
results_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "npe_a")
trajectories_path = str(path_to_repo / "results" / "real_examples" / "gipps_7d" / "trajectories")


def main(al, bl, Vl, xl0, vl0, p_accel, p_brake, tau, N):
    print("Generating xl, vl:")
    xl, vl = simulate_leader_trajectory(al, bl, Vl, xl0, vl0, p_accel, p_brake, tau, N)
    print("xl, vl generated successfully.")

    # Find next ID
    i = 0
    while os.path.exists(trajectories_path + f"/leader_trajectory{i}.npz"):
        i += 1

    config = {
            "al": al,
            "bl": bl,
            "Vl": Vl,
            "xl0": xl0,
            "vl0": vl0,
            "p_accel": p_accel,
            "p_brake": p_brake,
            "tau": tau,
            "N": N,
        }

    leader_trajectory_save_path = trajectories_path + f"/leader_trajectory{i}" + ".npz"
    config_save_path = trajectories_path + f"/leader_trajectory{i}" + ".yaml"
    
    print(f"Saving xl, vl to path {leader_trajectory_save_path}:")
    np.savez(leader_trajectory_save_path, xl=xl, vl=vl)
    print("xl, vl saved succesfully.")

    print(f"Saving config file to path {config_save_path}:")
    with open(config_save_path, "w") as f:
        yaml.safe_dump(config, f)
    print("Config file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--al", type=float, default=2.)
    parser.add_argument("--bl", type=float, default=-4.)
    parser.add_argument("--Vl", type=float, default=30.)
    parser.add_argument("--xl0", type=float, default=0.)
    parser.add_argument("--vl0", type=float, default=10.)
    parser.add_argument("--p_accel", type=float, default=0.2)
    parser.add_argument("--p_brake", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=200)
    args = parser.parse_args()
    main(args.al, args.bl, args.Vl, args.xl0, args.vl0, args.p_accel, args.p_brake, args.tau, args.N)
