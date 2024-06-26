import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

from distributions import Cauchy, Gaussian, Gamma, Laplace, T
from kernels import *
from discrepancies import MaximumMeanDiscrepancy, KernelSteinDiscrepancy

import sympy as sp

#mu = 0, sigma^2  = 1
mean_vec = np.zeros((1, 1))
cov_matrix = np.eye(1)
n = 3

inverse_multi_quadratic_kernel = InverseMultiQuadraticKernel(c=10, beta=-0.5)
gaussian_kernel = GaussianKernel(sigma=0.01)

gaussian = Gaussian(
    mu=np.zeros((1, 1)),
    covariance=np.eye(1),
)
t = T(degrees_of_freedom=1, loc=0, scale=1)
gamma = Gamma(k=1, theta=1)

stein_kernel = SteinKernel(
    kernel=gaussian_kernel,
    distribution=gaussian
)
p = np.asarray([1.])
q = np.asarray([4.])
# print(stein_kernel.kernel(p, q))
print(stein_kernel.k(p, q))

x = sp.symbols('x')
density = sp.exp(-(x**2) / 2)
DSOstein_kernel = DSOSteinKernel(
    kernel=gaussian_kernel,
    distribution=density
)

print(DSOstein_kernel.k(p, q))

# Y = np.random.multivariate_normal(np.zeros((1, 1)).flatten(), np.eye(1), 1)
# print(Y)
# print(gaussian.score(Y))
