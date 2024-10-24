import numpy as np
import csv
import os

N = 1000
mu = 0.0 #mean
var = 0.5 #sigma^2
a = 0.0
b = 1.0
scale = 1
# Step 1: Generate samples
samples = np.random.normal(mu, np.sqrt(var), N)
# samples = np.random.uniform(a, b, N)
# samples = np.random.beta(a, b, N)
# samples = np.random.exponential(scale, N)

# Step 2: Create pairs (x_i, x_i)
pairs = [(x, x) for x in samples]

# Specify the directory where you want to save the CSV file
directory = '/home/pl61/density-regression/deep-symbolic-optimization/dso/data'  # Change this to your desired directory

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Combine the directory path with the file name
# CHANGE THE NAME
file_path = os.path.join(directory, 'N(0,0.5)1000.csv')

# Step 3: Store the pairs in a CSV file
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['x_i', 'x_i_duplicate'])  # Write header
    writer.writerows(pairs)  # Write data

print(f"CSV file '{file_path}' created successfully.")
