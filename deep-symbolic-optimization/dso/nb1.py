import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

from distributions import Cauchy, Gaussian, Gamma, Laplace, T
from kernels import (
    PolynomialKernel,
    GaussianKernel,
    LaplacianKernel,
    InverseMultiQuadraticKernel,
    SteinKernel,
)
from discrepancies import MaximumMeanDiscrepancy, KernelSteinDiscrepancy

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
    distribution=gamma,
)

ksd = KernelSteinDiscrepancy(stein_kernel=stein_kernel)

# Y = np.random.multivariate_normal(np.zeros((1, 1)).flatten(), np.eye(1), n)
# print(Y)
for _ in range(3):
    Y = np.random.multivariate_normal(np.zeros((1, 1)).flatten(), np.eye(1), n)
    print(Y)
    # y = [1.,2.,3.,4.,5.]
    # print(y)
    # array = np.array([[ 0.42997167],
    #               [-0.34458528],
    #               [ 1.35881478],
    #               [-0.36571543],
    #               [-0.95776895]])
    print(ksd.compute(Y))
    # print(ksd.compute(np.array(y).reshape(-1, 1))) 
    # #Y is iid p samples, and q is the distribution parameter in the stein kernel initialization
    