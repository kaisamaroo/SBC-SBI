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
