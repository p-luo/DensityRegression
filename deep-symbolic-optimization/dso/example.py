import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

from dso.distributions import Gaussian
from dso.kernels import *
from dso.discrepancies import MaximumMeanDiscrepancy, KernelSteinDiscrepancy, DSOKernelSteinDiscrepancy

import sympy as sp

#initializing parameters of normal distribution
mean_vec = np.zeros((1, 1))
cov_matrix = np.eye(1)

#defining some kernels, from distributiondiscrepancies/kernels.py
inverse_multi_quadratic_kernel = InverseMultiQuadraticKernel(c=10, beta=-0.5)
gaussian_kernel = GaussianKernel(sigma=0.01)

#Defining the Gaussian distribution from distributiondiscrepancies/distributions.py
gaussian = Gaussian(
    mu=np.zeros((1, 1)),
    covariance=np.eye(1),
)

#Initializing the Stein Kernel from kernels.py
stein_kernel = SteinKernel(
    kernel=gaussian_kernel,
    distribution=gaussian
)

#This evaluates u_q(a,b), where the Stein Discrepancy S(p,q)=E_{x,y~p}[u_q(x,y)]
a = np.asarray([1.])
b = np.asarray([4.])
print(stein_kernel.k(a, b)) 

#Setting up Gaussian density; the normalization constant doesn't matter in the computation
x = sp.symbols('x')
density = sp.exp(-(x**2) / 2) 

#Defining the DSO-compatible Stein Kernel, which takes as parameters a kernel from kernels.py, and a sympy expression (distribution) 
DSOstein_kernel = DSOSteinKernel(
    kernel=gaussian_kernel,
    distribution=density
)

print(DSOstein_kernel.k(a, b)) #Should print the same thing

#Defining Discrepancy objects and obtaining unbiased estimates of E_{x,y~p}[u_q(x,y)]
print("Unbiased estimates of E_{x,y~p}[u_q(x,y)]:")
data = np.asarray([1., 2., 3., 4.]).reshape(-1, 1)
s = KernelSteinDiscrepancy(stein_kernel = stein_kernel)
print("Stein Kernel from the package")
print(s.compute(data))

ss = DSOKernelSteinDiscrepancy(DSO_stein_kernel = DSOstein_kernel)
print("DSO Stein Kernel, with Sympy compatibility")
print(ss.compute(data)) #Should print the same thing
