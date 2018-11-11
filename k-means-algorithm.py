# Andres Gandara
# CSE 494 - Li
# Introduction to Machine Learning
# Clustering with K-Means Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize dataset
dataset = pd.DataFrame({
    'x': [2, 3, 1, 5, 7, 6, 8, 4],
    'y': [8, 3, 2, 8, 3, 4, 4, 7]
})

# Using k of 3
k = 3

# Initial centers (one for each k) for 1.1, 1.2, and 1.3
initCenters_1 = { 
    1: [2, 8], 
    2: [3, 3], 
    3: [5, 8]
}
initCenters_2 = { 
    1: [2, 8], 
    2: [3, 3], 
    3: [6, 4] 
}
initCenters_3 = { 
    1: [2, 8], 
    2: [1, 2], 
    3: [6, 4] 
}

# Define which set of centers you want to use
initCenters = initCenters_1
print('Using the following initial centers:')
for p in initCenters.keys():
    print(initCenters[p])

# Plot initial dataset and centers
fig = plt.figure(figsize=(5, 5))
plt.scatter(dataset['x'], dataset['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in initCenters.keys():
    plt.scatter(*initCenters[i], color=colmap[i])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Initial Centers & Dataset')
plt.show()

# Classify function
def classify(dataset, initCenters):
    for i in initCenters.keys():
        dataset['dist_from_{}'.format(i)] = (
            np.sqrt(
                (dataset['x'] - initCenters[i][0]) ** 2 + 
                (dataset['y'] - initCenters[i][1]) ** 2
            )
        )
    initCenters_dist_cols = ['dist_from_{}'.format(i) for i in initCenters.keys()]
    dataset['closest'] = dataset.loc[:, initCenters_dist_cols].idxmin(axis=1)
    dataset['closest'] = dataset['closest'].map(lambda x: int(x.lstrip('dist_from_')))
    dataset['color'] = dataset['closest'].map(lambda x: colmap[x])
    return dataset

# Calculate distances for initCenters
dataset = classify(dataset, initCenters)
print(dataset.head())


# Update centers function
def update(x):
    for i in initCenters.keys():
        initCenters[i][0] = np.mean(dataset[dataset['closest'] == i]['x'])
        initCenters[i][1] = np.mean(dataset[dataset['closest'] == i]['y'])
    return x

# Perform update and classification until no changes are detected
while True:
    nearestCenters = dataset['closest'].copy(deep = True)
    initCenters = update(initCenters)
    dataset = classify(dataset, initCenters)
    if nearestCenters.equals(dataset['closest']):
        break

# Display final clusters
fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
plt.scatter(dataset['x'], dataset['y'], color=dataset['color'], alpha=0.4, edgecolor='k')
for i in initCenters.keys():
    plt.scatter(*initCenters[i], color=colmap[i])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Final Clusters')
plt.show()