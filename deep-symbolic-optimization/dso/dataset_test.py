import numpy as np
from dso.distributions import Gaussian
from dso.kernels import *
from dso.discrepancies import DSOKernelSteinDiscrepancy

import sympy as sp

gaussian_kernel = GaussianKernel(sigma=0.01) 

x = sp.symbols('x')
# density = sp.exp(sp.log(sp.log(x))**2/x)
# density = sp.exp(x)*sp.log(sp.log(x**2 + x - sp.log(sp.log(sp.log(sp.log(x))))))
density = sp.log(x + sp.exp(x**2))
# density = sp.log(sp.log(x*(-x + sp.log(x))))
# density = sp.log(sp.log(x**2/sp.log(2*x) - x))
# density = sp.exp(-(x**2) / 2) 

#Defining the DSO-compatible Stein Kernel, which takes as parameters a kernel object from kernels.py, and a sympy expression (the PDF) 
DSOstein_kernel = DSOSteinKernel(
    kernel=gaussian_kernel,
    distribution=density
)

N = 200
iters = 20
num = 20
reward = []
ss = DSOKernelSteinDiscrepancy(DSO_stein_kernel = DSOstein_kernel)
for _ in range(num):
    reward = []
    for i in range(iters):
        samples = np.random.normal(0, 1, N)
        data = np.asarray(samples).reshape(-1,1)
        cur = ss.compute(data)
        reward.append(1/(1+np.abs(cur)))
    print(np.mean(reward))
