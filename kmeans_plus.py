# Template program.
# Input: datafile, k (#clusters), r (#iterations)
# Each row in datafile is a point. k is an integer. r is an integer.
# Output: cluster number of each data points

import os
import sys

import numpy as np


def initialize_clusters(data, k):
    centroids = [data[0]]
    for k in range(1, k):
        temp = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
        probability = temp / temp.sum()
        cumulative_probability = probability.cumsum()
        r = np.random.rand()
        x = 0
        for j, p in enumerate(cumulative_probability):
            if r < p:
                x = j
                break
        centroids.append(data[x])
    return np.array(centroids)


def closest_centroid(points, centroids):
    # returns an array containing the index to the nearest centroid for each point
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    # returns the new centroids assigned from the points closest to them
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


def extract_data(cwd, data_file):
    # Returns the data fron the file and changing the values to float
    data = np.loadtxt(os.path.join(cwd, data_file), delimiter=',')
    return np.array(data)


def kmeanspp(points, centroids, max_iter):
    closest = []
    for i in range(max_iter):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    return closest, centroids


def error(data, predicted_clusters, clusters):
    error = 0
    for index, cluster in enumerate(predicted_clusters):
        error += np.linalg.norm(data[index] - clusters[cluster]) ** 2
    return error


if __name__ == '__main__':

    np.random.seed(np.random.randint(0, 5))
    args = sys.argv

    if len(args) != 5:
        print('usage: ', sys.argv[0], 'data_file k r output_file')
        sys.exit()

    data = extract_data(os.getcwd(), args[1])

    k = int(args[2])
    max_iter = int(args[3])
    output_file = args[4]

    points = extract_data(os.getcwd(), args[1])
    centroids = initialize_clusters(points, k)
    closest, final_centroids = kmeanspp(points, centroids, max_iter)

    print(closest)
    with open(output_file, 'w') as f:
        for item in closest:
            f.write("%s\n" % item)

    print(error(points, closest, final_centroids))
