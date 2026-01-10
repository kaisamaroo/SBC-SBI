Master's project investigating the ability of simulation-based calibration (SBC) to validate modern simulation-based inference (SBI) methods.

We apply modern SBI methods to a range of examples, consisting of both toy and real-world simulators. See below for a directory of the examples in this repository:

## Toy examples

### **norm_norm_diffuse_1d**: Diffuse 1-dimensional Gaussian prior with Gaussian likelihood.

Given a (large) standard deviation $\sigma$ (we take the default to be $150$), we define the following diffuse Gaussian prior centred at $0$:

$$p(\theta) = N(\theta ; 0, \sigma^2)$$

We define our simulator as follows

$$p(x|\theta) = N(x; \theta , 1) \hspace{5mm} \theta \in \mathbb{R}$$

Suppose we observe some data $x_\text{observed} \in \mathbb{R}$. In this case, we have a true posterior given analytically by

$$p(\theta|x_\text{observed}) = N\left(\theta; \frac{\sigma^2}{1+\sigma^2} x_\text{observed}, \frac{\sigma^2}{1+\sigma^2}\right)$$

Since $\sigma$ is large, the true posterior is much narrower than the prior (variance of posterior = $\frac{\sigma^2}{1+\sigma^2} \to 1$ as $\sigma^2 \to \infty$). Thus, the posterior mass intersects the prior mass in only a small region. This example aims to mimic what is commonly done in practise where practitioners select very diffuse priors. We hypothesize that it will be hard for SBI methods to recover a good posterior approximation.

## Real examples

### **gipps_7d**: Parameter inference on the Gipps car following model.

In this experiment, we use SBC to validate SBI methods applied to the Gipps car following model (original paper: https://www.sciencedirect.com/science/article/pii/0191261581900370). We follow the experiments used in https://www.sciencedirect.com/science/article/pii/S0378437124001808?fr=RR-2&ref=pdf_download&rr=9a4270eb8b679466, who apply Bayesian inference (MCMC) to infer the parameters of the Gipps ODE that generated a given follower trajectory. Instead of MCMC, we apply SBI to the inference problem, and then validate these trained SBI models using SBC. No code was provided by the authors, so a direct comparison to their methodology is not possible.

The Gipps model is a (non-stochastic) differential equation that can be solved to obtain the displacement time series $\{x^\text{Gipps}_f(t) : t \in [0,T]\}$ of a follower car, given a leader car's trajectory $\{(x_l(t), v_l(t)) : t \in [0,T]\}$ and some driver-specific ODE parameters $(a_f, b_f, V_f, x_f(0), v_f(0), \tau, \ell_l, \Psi, b_l)$, where

- $a_f$: Maximum acceleration of the follower vehicle.

- $b_f$: Maximum deceleration (braking) of the follower vehicle. Note that $b_f < 0$.

- $V_f$: Desired maximum velocity of the follower vehicle. If in free-flow (no congestion), the driver will accelerate to $V_f$ and remain at this velocity indefinitely. 

- $x_f(0)$: Initial position of the follower vehicle.

- $v_f(0)$: Initial velocity of the follower vehicle. 

- $\tau$: Driver reaction time (the time delay before the follower responds to changes in the leader’s motion).

- $\ell_l$: Effective length of the leader vehicle.

- $\Psi$: Anticipation factor representing how well the follower intereprets the leader’s maximum braking. In particular, the follower's best guess of the leaders maximum braking is $\Psi b_l$.

- $b_l$: Maximum deceleration of the leader vehicle.

In applications, the differential equation is discretised into $N+1$ discrete time points to become a difference equation on the grid $\{0, \frac{T}{N}, ..., \frac{T(N-1)}{N}, T\}$. Interestingly, practitioners often take the discretization gap $\frac{T}{N}$ to be the driver's reaction time $\tau$, which makes sense intuitively, but makes Bayesian inference on $\tau$ impossible (since $\tau$ is the discretization width of the data, so must be known during data collection). For this reason, we hold $\tau$ constant throughout the problem (we take the default to be $0.5s$, in-line with reality). We also hold constant the following variables to reduce the number of variables to infer:

- The maximum deceleration $b_l$ of the leader vehicle (default $-4ms^{-1}$ )

- The anticipation factor $\Psi$ (default $1.05$).

- The effective length $\ell_l$ of the leader vehicle (default $7.5m$).

For $t \in \{0, \frac{T}{N}, ..., \frac{T(N-1)}{N}\}$, the difference equation is

$$v_f(t + \tau) = \min\!\big( v_{ff}(t + \tau)\ v_{cf}(t + \tau) \big)$$

$$v_{ff}(t + \tau)  = v_f(t)  + 2.5\ a_{f}\ \tau  \left( 1 - \frac{v_f(t)}{V_{f}} \right) \sqrt{0.025 + \frac{v_f(t)}{V_{f}}}$$

$$v_{cf}(t + \tau)
= b_{f}\ \tau + \sqrt{b_{f}^2 \tau^2 - b_{f} \left[ 2\big(x_l(t) - \ell_l - x_f(t)\big) - v_f(t)\ \tau - \frac{v_l^2(t)}{\Psi b_l} \right] }$$

Given 

- The discretized leader trajectory $\{(x_l(t), v_l(t)) : t \in \{0, \tau, 2\tau, ..., T-\tau, T\}\}$
- Parameters $(a_f, b_f, V_f, b_l, x_f(0), v_f(0))$ (and $(\tau, \ell_l, \Psi, b_l)$, but these are fixed throughout)

we can use the difference equation above to compute (deterministically) the follower trajectory 

$$\{(x^\text{Gipps}_f(t), v^\text{Gipps}_f(t)) : t \in \{0, \tau, 2\tau, ..., T-\tau, T\}\}$$

In pseudocode, we can express the Gipps ODE solver as:

$$x^\text{Gipps}_f, v^\text{Gipps}_f = \texttt{gipps}(a_f, b_f, V_f, x_f(0), v_f(0), x_l, v_l)$$

Note that solving the Gipps equation gives the follower velocity time series, but using the first-order approximation $x^\text{Gipps}_f(t + \tau) = x^\text{Gipps}_f(t) + \frac{1}{2} \tau (v^\text{Gipps}_f(t) + v^\text{Gipps}_f(t + \tau)))$ we obtain the follower displacement time series and therefore the full follower trajectory (we define a trajectory as the displacement and velocity time series). Since our analysis depends only on the follower's displacement time series, we discard the velocity time series after this step.

To define our simulator, we further account for noise in the measurement devices used to observe the follower displacement time series. Specifically, we add Gaussian noise with mean $\mu \in \mathbb{R}^N$ and covariance $\sigma^2 I_{N \times N}$ to the Gipps displacement time series ${x^\text{Gipps}_f(t) : t \in {0, \tau, 2\tau, \ldots, T-\tau, T}}$, yielding the “stochastic” follower displacement time series ${x_f(t) : t \in {0, \tau, 2\tau, \ldots, T-\tau, T}}$. The sole source of stochasticity in the simulator is this Gaussian measurement noise. To be concrete, we wish to use SBI to infer the parameters $(a_f, b_f, V_f, x_f(0), v_f(0), \mu, \sigma^2)$, whereas we hold fixed the parameters $(N, T, \tau, b_l, \Phi, \ell_l)$. We can express our simulator in pseudocode as

$$x_f = \texttt{simulator}(a_f, b_f, V_f, x_f(0), v_f(0), \mu, \sigma^2, x_l, v_l)$$

In our experiments, we assumed the leader car follows the Gipps’ free flow component ($v_{ff}$ above), with random acceleration/deceleration perturbations imposed at each time step to simulate typical driving conditions (i.e. the leader will accelerate/brake accoriding to its leader, etc...). 

We place priors over the parameters $(a_f, b_f, V_f, x_f(0), v_f(0), \mu, \sigma^2)$ we wish to infer, and apply various SBI methods to retrieve the posterior $p\left(a_f, b_f, V_f, x_f(0), v_f(0), \mu, \sigma^2 | x_f ; x_l, v_l\right)$.
