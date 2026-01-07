Master's project investigating modern simulation-based inference (SBI) methods using simulation-based calibration (SBC).

We apply modern SBI methods to a range of examples, consisting of both toy and real-world simulators. See below for a directory of the examples in this repository:

## Toy examples

### **norm_norm_diffuse_1d**: Diffuse 1-dimensional Gaussian prior with Gaussian likelihood.

Given a (large) standard deviation $\sigma$ (we take the default to be $150$), we define the following diffuse Gaussian prior centred at $0$:

$$p(\theta) = N(\theta ; 0, \sigma^2)$$

We define our simulator as follows

$$p(x|\theta) = N(x; \theta , 1) \hspace{5mm} \theta \in \mathbb{R}$$

Suppose we observe some data $x_\text{observed} \in \mathbb{R}$. In this case, we have a posterior given analytically by

$$p(\theta|x_\text{observed}) = N\left(\theta; \frac{\sigma^2}{1+\sigma^2} x_\text{observed}, \frac{\sigma^2}{1+\sigma^2}\right)$$

This example aims to mimic what is commonly done in practise where practitioners select very diffuse priors. We hypothesise that it will be hard for SBI to recover a good posterior approximation.

## Real examples

### **gipps_7d**: Parameter inference on the Gipps car following model.

In this experiment, we apply SBI and SBC to the Gipps car following model (original paper: https://www.sciencedirect.com/science/article/pii/0191261581900370). We follow the experiments used in https://www.sciencedirect.com/science/article/pii/S0378437124001808?fr=RR-2&ref=pdf_download&rr=9a4270eb8b679466, who apply Bayesian inference (MCMC) to estimate the parameters of the Gipps ODE that generated a given follower trajectory. We tweak this paper by applying SBI and SBC to the inference problem. No code was provided by the authors, so a direct comparison to their methodology is not possible.

The Gipps model is a differential equation that deterministically defines the displacement $\{x_f(t) : t \in [0,T]\}$ of a follower car, given a leader's trajectory $\{(x_l(t), v_l(t)) : t \in [0,T]\}$ and some driver-specific ODE parameters $(\tau, a_f, b_f, V_f, \ell_l, b_l, \Psi, x_f(0), v_f(0))$, where

- $\tau$: Driver reaction time (the time delay before the follower responds to changes in the leader’s motion).

- $a_f$: Maximum acceleration of the follower vehicle.

- $b_f$: Maximum deceleration (braking) of the follower vehicle. Note that $b_f < 0$.

- $V_f$: Desired maximum velocity of the follower vehicle. If in free-flow (no congestion), the driver will accelerate to $V_f$ and remain at this velocity indefinitely.

- $\ell_l$: Effective length of the leader vehicle.

- $b_l$: Maximum deceleration of the leader vehicle.

- $\Psi$: Anticipation factor representing how well the follower intereprets the leader’s maximum braking. In particular, the follower's best guess of the leaders maximum braking is $\Psi b_l$.

- $x_f(0)$: Initial position of the follower vehicle.

- $v_f(0)$: Initial velocity of the follower vehicle.

In applications, the differential equation is discretised to become a difference equation. Interestingly, the discretization gap is chosen to be the driver's reaction time $\tau$, which makes sense intuitively, but makes Bayesian inference on $\tau$ impossible (since $\tau$ is the discretization width of the data, so it must be fixed and cannot be inferred). The difference equation is

$$v_f(t + \tau) = \min\!\big( v_{ff}(t + \tau),\, v_{cf}(t + \tau) \big)$$


$$v_{ff}(t + \tau) 
= v_f(t) 
+ 2.5\, a_{f}\, \tau 
\left( 1 - \frac{v_f(t)}{V_{f}} \right)
\sqrt{0.025 + \frac{v_f(t)}{V_{f}}}$$

$$v_{cf}(t + \tau)
= b_{f}\, \tau
+ \sqrt{
b_{f}^2 \tau^2
- b_{f}
\left[
2\big(x_l(t) - \ell_l - x_f(t)\big)
- v_f(t)\, \tau
- \frac{v_l^2(t)}{\Psi b_l}
\right]
}$$

Given 

- The discretized leader trajectory $\{(x_l(t), v_l(t)) : t \in \{0, \tau, 2\tau, ..., T-\tau, T\}\}$
- Hyperparameters $(\tau, a_f, b_f, V_f, \ell_l, b_l, \Psi, x_f(0), v_f(0))$

we can use the difference equation above to simulate the follower trajectory 

$$\{(x_f(t), v_f(t)) : t \in \{0, \tau, 2\tau, ..., T-\tau, T\}\}$$

Note that solving the Gipps equation gives the follower velocity time series, but using the approximation $x_f(t + \tau) = x_f(t) + \frac{1}{2} \tau (v_f(t) + v_f(t + \tau))$) we obtain the follower displacement and therefore the full follower trajectory.
