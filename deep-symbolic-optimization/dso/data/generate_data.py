import numpy as np
import csv
import os

N = 500
# Step 1: Generate samples
samples = np.random.normal(0.0, 1.0, N)
# samples = np.random.uniform(0.0, 1.0, N)
# samples = np.random.beta(2.0, 5.0, N)

# Step 2: Create pairs (x_i, x_i)
pairs = [(x, x) for x in samples]

# Specify the directory where you want to save the CSV file
directory = '/home/pl61/density-regression/deep-symbolic-optimization/dso/data'  # Change this to your desired directory

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Combine the directory path with the file name
file_path = os.path.join(directory, 'N(0,1)500.csv')

# Step 3: Store the pairs in a CSV file
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['x_i', 'x_i_duplicate'])  # Write header
    writer.writerows(pairs)  # Write data

print(f"CSV file '{file_path}' created successfully.")
