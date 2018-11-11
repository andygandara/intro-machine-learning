# Andres Gandara
# CSE 494 - Li
# Introduction to Machine Learning
# Principal Component Analysis

import numpy as np
import matplotlib.pyplot as plt

# Parse data file
data = {
    'x': [],
    'y': []
}
with open("pca_data.txt") as f:
    for line in f:
        (x, y) = line.split()
        data['x'].append(float(x))
        data['y'].append(float(y))
#        data.append([float(x), float(y)])

print(data['x'])
print(data['y'])

# Plot dataset
fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'])
plt.xlim(2, 9)
plt.ylim(-2, 9)
plt.title('Data')
plt.show()

    
# Covariance function
# Takes 2 arrays for x and y values
def covar(x, y):
    # Find means
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    # Length of x and y
    n = len(x)
    
    # Find summation
    topSum = 0.0
    for i in range(n):
        topSum += (x[i] - mean_x) * (y[i] - mean_y)
        
    return topSum / (n - 1)

# Variance function
# Takes an array of values
def var(z):
    return covar(z, z)

# Create covariance matrix
cov_matrix = [
    [var(data['x']), covar(data['x'], data['y'])],
    [covar(data['x'], data['y']), var(data['y'])]
]
print('Covariance Matrix: \n%s\n' %cov_matrix)

# Find Eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigenvectors: \n%s\n' %eig_vecs)
print('Eigenvalues: \n%s\n' %eig_vals)

    
    
    
    
    
    
    
    
    
    
    
    
    
    