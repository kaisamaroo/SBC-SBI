Master's project investigating modern simulation-based inference (SBI) methods using simulation-based calibration (SBC).

We apply modern SBI methods to a range of examples, consisting of both toy and real-world simulators. See below for a directory of the examples in this repository:

## Toy examples

### **norm_norm_diffuse_1d**:

We hypothesize that SNPE-A will struggle to provide a good posterior approximation when the prior (or proposal) mass is far from the posterior mass. 

We can test this in 1D by considering the following example:

$$p(\theta) = N(\theta ; 0, 1)$$

$$p(x|\theta) = N(x; \theta , 1)$$

where we observe $x_\text{obs}\in \{0,1,2,3,...\}$ ($x_\text{obs}$ is $x_\text{obs}$ standard deviations above the mean of the prior). In this case, we have a posterior given analytically by

$$p(\theta|x_\text{obs}) = N\left(\theta; \frac{x_\text{obs}}{2}, \frac{1}{2}\right)$$

We expect that, for large enough $x$ (think over $\approx 4\sigma$ away from prior mean), the SNPE-A algorithm will struggle to approximate the true posterior, since it will be extrapolating past the $(x, \theta)$ training pairs. 

Note that the SNPE-A algorithm (and therefore the python implementation from `sbi`) only allows Gaussian or uniform priors.
