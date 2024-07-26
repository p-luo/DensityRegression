import numpy as np

from dso.distributions import Gaussian
from dso.kernels import *
from dso.discrepancies import DSOKernelSteinDiscrepancy

import sympy as sp
from scipy import stats

gaussian_kernel = GaussianKernel(sigma=0.01) 

x = sp.symbols('x')
# A few densities that were outputted by the DSO algorithm. Some of them return nan values.
# density = sp.exp(sp.log(sp.log(x))**2/x)
# density = sp.exp(x)*sp.log(sp.log(x**2 + x - sp.log(sp.log(sp.log(sp.log(x))))))
# density = x**2 + (x*sp.exp(x))/(x**2+sp.exp(1)*x)
density = 1.5-sp.exp(-x**2+0.1*x)
# density = sp.log(sp.log(x*(-x + sp.log(x))))
# density = sp.log(sp.log(x**2/sp.log(2*x) - x))
# density = sp.exp(-(x**2) / 2) 
# density = sp.cos(x)

#Defining the DSO-compatible Stein Kernel, which takes as parameters a kernel object from kernels.py, and a sympy expression (the PDF) 
DSOstein_kernel = DSOSteinKernel(
    kernel=gaussian_kernel,
    distribution=density
)

def reward_function(s):
    return 0 if s <= -2 else 1/(1+np.abs(s))
    # return 1/(1+np.abs(s))

def aggregate(arr):
    # return np.median(arr)
    return np.mean(arr)

N = 200 #Size of dataset
iters = 100 #Number of times that data is sampled and then an estimate of S(p,q) is computed. Then it computes and records the reward function. After the inner for loop, it outputs the average reward.
num = 1 #Number of times that the above algorithm is run
reward = [] 
rewards = []
ss = DSOKernelSteinDiscrepancy(DSO_stein_kernel = DSOstein_kernel)
for _ in range(num):
    reward = []
    for i in range(iters):
        samples = np.random.normal(0, 1, N)
        data = np.asarray(samples).reshape(-1,1)
        cur = ss.compute(data)
        # print(cur) #Will output the estimated Stein discrepancy, using samples
        reward.append(reward_function(cur))
    # rewards.append(aggregate(reward))
    # print(aggregate(reward))

# print(reward)
print(stats.describe(reward))
