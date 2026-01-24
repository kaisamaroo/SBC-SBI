import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, InverseGamma, MultivariateNormal
from sbi.utils import MultipleIndependent, BoxUniform


def one_step_update(af, bf, Vf, tau, ll, psi, vf_prev, xf_prev, vl_prev, xl_prev, bl):
    """
    Perform a one-step (i.e one reaction timestep) update on the velocity of the follower
    """
    vff = max(0, vf_prev + 2.5 * af * tau * (1 - vf_prev / Vf) * np.sqrt(max(0, 0.025 + vf_prev / Vf)))
    vcf = max(0, bf * tau + np.sqrt(max(0, bf**2 * tau**2 - bf * (2 * (xl_prev - ll - xf_prev) - vf_prev * tau - (vl_prev**2)/(psi * bl)))))
    return min(vff, vcf)


def follower_trajectory(af, bf, Vf, tau, ll, psi, vf0, xf0, vl, xl, N, bl):
    """
    Generate entire follower trajectory (xf, vf) given entire leader 
    trajectory (xl, vl) and relevant Gipps parameters.
    """
    xf = [xf0]
    vf = [vf0]
    xf_prev = xf0
    vf_prev = vf0
    for step in range(1, N+1):
        xl_prev = xl[step - 1]
        vl_prev = vl[step - 1]
        vf_new = one_step_update(af, bf, Vf, tau, ll, psi, vf_prev, xf_prev, vl_prev, xl_prev, bl)
        xf_new = xf_prev + tau * (vf_prev + vf_new)/2
        xf.append(xf_new)
        vf.append(vf_new)
        xf_prev = xf_new
        vf_prev = vf_new
    return xf, vf


def simulate_leader_trajectory(al, bl, Vl, xl0, vl0, p_accel, p_brake, tau, N):
    """
    Generate a stochastic leader trajectory that follows the free-flow part of Gipps
    ODE, but add random accelerating and braking with probabilities p_accel and 
    p_brake to simulate unpredictable leader behaviour.
    """
    xl = [xl0]
    vl = [vl0]
    xl_prev = xl0
    vl_prev = vl0
    for step in range(1, N+1):

        u = np.random.rand()
        if u < p_accel:
            vl_new = max(0, vl_prev + al * tau)
        elif u < p_accel + p_brake:
            vl_new = max(0, vl_prev + bl * tau)
        else:
            vl_new = max(0, vl_prev + 2.5 * al * tau * (1 - vl_prev / Vl) * np.sqrt(max(0, 0.025 + vl_prev / Vl)))
        vl.append(vl_new)
        xl_new = xl_prev + tau * (vl_prev + vl_new)/2
        xl.append(xl_new)
        xl_prev = xl_new
        vl_prev = vl_new
    return xl, vl


def plot_xf_and_xl(xf, xl, tau, N):
    fig, ax = plt.subplots(figsize=(10,5))
    t_range = tau * np.arange(0, N+1)
    ax.plot(t_range, xf, color="blue", label=r"Displacement of follower $x_f$")
    ax.plot(t_range, xl, color="red", label=r"Displacement of leader $x_l$")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Displacement")
    if N < 250:
        ax.set_xticks(t_range, minor=True)
    plt.legend()
    plt.show()

def plot_diff_xf_and_xl(xf, xl, tau, N, ll):
    fig, ax = plt.subplots(figsize=(10,5))
    t_range = tau * np.arange(0, N+1)
    ax.plot(t_range, np.array(xl) - np.array(xf) - ll, color="black", label=r"Distance between follower and leader $x_l - x_f - \ell_l$")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Distance between follower and leader")
    if N < 250:
        ax.set_xticks(t_range, minor=True)
    plt.legend()
    plt.show()

def plot_vf_and_vl(vf, vl, tau, N):
    fig, ax = plt.subplots(figsize=(10,5))
    t_range = tau * np.arange(0, N+1)
    ax.plot(t_range, vf, color="blue", label=r"Velocity of follower $v_f$")
    ax.plot(t_range, vl, color="red", label=r"Velocity of leader $v_l$")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Velocity")
    if N < 250:
        ax.set_xticks(t_range, minor=True)
    plt.legend()
    plt.show()

def plot_all(xf, vf, xl, vl, tau, N, ll):
    fig, ax = plt.subplots(figsize=(10,10), nrows=3, ncols=1)
    t_range = tau * np.arange(0, N+1)

    ax[0].plot(t_range, xf, color="blue", label=r"Displacement of follower $x_f$")
    ax[0].plot(t_range, xl, color="red", label=r"Displacement of leader $x_l$")
    ax[0].set_xlabel(r"Time (s)")
    ax[0].set_ylabel("Displacement (m)")
    ax[0].legend()

    ax[1].plot(t_range, np.array(xl) - np.array(xf) - ll, color="black", label=r"Distance between leader and follower $x_l - x_f - \ell_l$")
    ax[1].set_xlabel(r"Time (s)")
    ax[1].set_ylabel("Distance between leader and follower (m)")
    ax[1].legend()

    ax[2].plot(t_range, vf, color="blue", label=r"Velocity of follower $v_f$")
    ax[2].plot(t_range, vl, color="red", label=r"Velocity of leader $v_l$")
    ax[2].set_xlabel(r"Time (s)")
    ax[2].set_ylabel(r"Velocity $(ms^{-1})$")
    ax[2].legend()

    if N < 250:
        ax[0].set_xticks(t_range, minor=True)
        ax[1].set_xticks(t_range, minor=True)
        ax[2].set_xticks(t_range, minor=True)

    plt.tight_layout()
    plt.show()


def follower_trajectory_stochastic(af, bf, Vf, xf0, vf0, mu, sigmasquared, tau, N, ll, psi, xl, vl, bl):
    """
    af, bf, Vf, xv0, vf0 are the BAYESIAN PARAMETERS. 
    All other arguments are fixed.
    """
    xf, vf = follower_trajectory(af, bf, Vf, tau, ll, psi, vf0, xf0, vl, xl, N, bl)
    xf = torch.tensor(xf) 
    noise = torch.distributions.MultivariateNormal(mu*torch.ones(N+1), covariance_matrix=sigmasquared*torch.eye(N+1)).sample()
    return xf + noise, torch.tensor(vf) # Returns torch.tensors


def simulator(theta, tau, N, ll, psi, xl, vl, bl):
    """
    theta can be either:

    a 1d numpy array of the form array([af, bf, Vf, xf0, vf0, mu, sigmasquared])
    OR
    a 2d numpy array that contains multiple [af, bf, Vf, xf0, vf0, mu, sigmasquared]
    """
    theta = theta.reshape(-1, 7)
    n_samples = theta.shape[0]
    simulations = torch.zeros((n_samples, N+1))
    for simulation in range(n_samples):
        af, bf, Vf, xf0, vf0, mu, sigmasquared = theta[simulation, :]
        simulated_data, _ = follower_trajectory_stochastic(af, bf, Vf, xf0, vf0, mu, sigmasquared, tau, N, ll, psi, xl, vl, bl)
        simulations[simulation, :] = simulated_data
    return simulations
    

def make_prior_7d_npe_a(aL, aU,
                    bL, bU,
                    VL, VU,
                    xf0L, xf0U,
                    vf0L, vf0U,
                    muL, muU,
                    sigmasquaredL, sigmasquaredU):

    return BoxUniform(
                        torch.tensor([aL, bL, VL, xf0L, vf0L, muL, sigmasquaredL]),
                        torch.tensor([aU, bU, VU, xf0U, vf0U, muU, sigmasquaredU])
                    )


def make_prior_7d_npe_c(aL, aU,
        bL, bU,
        VL, VU,
        xf0L, xf0U,
        vf0L, vf0U,
        prior_mean_mu, prior_variance_mu,
        prior_alpha_sigmasquared, prior_beta_sigmasquared):

    return MultipleIndependent([
        BoxUniform(torch.tensor([aL, bL, VL, xf0L, vf0L]),
                torch.tensor([aU, bU, VU, xf0U, vf0U])),
        Normal(torch.tensor([prior_mean_mu]), torch.tensor([np.sqrt(prior_variance_mu)])),
        InverseGamma(torch.tensor([prior_alpha_sigmasquared]), torch.tensor([1/prior_beta_sigmasquared])) # NOT SURE ABOUT SCALE OR RATE PARAM. IS BETA THE SCALE OR THE RATE?
    ])


# Projection test function
def test_function_projection(theta, i):
    """
    theta is a 1D or 2D torch.tensor 

    project theta onto its i'th dimension
    """
    if theta.dim() == 1:
        return theta[i]
    else:
        # In this case, theta is (batch_size, parameter_dimension)
        return theta[:, i]
    

def test_function_squared_norm(theta):
    """
    theta is a 1D or 2D torch.tensor 

    return squared norm of parameter vector
    """
    if theta.dim() == 1:
        return torch.sum(theta**2)
    else:
        # In this case, theta is (batch_size, parameter_dimension)
        return torch.sum(theta**2, axis=1)


# UPDATE THIS ALONG WITH all_test_function_names
def get_test_function(test_function_name="projection0"):
    for i in range(7):
        if test_function_name == f"projection{i}":
            return lambda x : test_function_projection(x, i)
    if test_function_name=="squared_norm":
        return test_function_squared_norm
    else:
        raise NotImplementedError("No matching test function.")
    
# UPDATE THIS ALONG WITH get_test_function
all_test_function_names = [f"projection{i}" for i in range(7)] \
                        + ["squared_norm"]
