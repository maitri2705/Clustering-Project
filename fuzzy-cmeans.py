import math
import os
import sys

import numpy as np
import numpy.linalg as LA

from kmeans_plus import initialize_clusters, extract_data, closest_centroid


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

    data = np.genfromtxt(args[1], delimiter=',')
    k = int(args[2])
    max_iter = int(args[3])
    output_file = args[4]

    points = extract_data(os.getcwd(), args[1])
    centroids = initialize_clusters(points, k)

    m = 1.5  # Fuzzy parameter (it can be tuned)
    r = (2 / (m - 1))
    size = len(data)

    ccentroids = [[] for x in range(k)]

    for j in range(max_iter):
        u = [[] for x in range(k)]

        for i in range(size):
            # Distances (of every point to each centroid)
            distance = [LA.norm(data[i] - centroids[x]) for x in range(k)]
            # Pertinence matrix vectors:
            U = []
            for a in distance:
                sum = 0
                for b in distance:
                    if b != 0:
                        c = a / b
                        sum += math.pow(c, r)
                if sum != 0:
                    U.append(1 / sum)
                else:
                    U.append(0)

            for x in range(k):
                # We will get an array of n row points x K centroids, with their degree of pertenence
                u[x].append(U[x])

        # now we calculate new centers
        centroids = [(np.array(u[i]) ** 2).dot(data) / np.sum(np.array(u[i]) ** 2) for i in range(k)]

        # save centroids
        for i in range(k):
            ccentroids[i].append(centroids[i])

        if j > k:
            changed_rate = [
                np.sum(
                    3 * ccentroids[x][j] - ccentroids[x][j - 1] - ccentroids[x][j - 2] - ccentroids[x][j - 3]) / 3
                for x in range(k)]
            change_rate = np.array(changed_rate)
            changed = np.sum(change_rate > 0.0000001)
            if changed == 0:
                break

    final_centroids = []
    j = np.array(ccentroids).shape[1]

    for i in range(k):
        final_centroids.append(ccentroids[i][j - 1])

    closest = closest_centroid(points, np.array(final_centroids))

    # print(closest)
    with open(output_file, 'w') as f:
        for item in closest:
            f.write("%s\n" % item)

    print(error(points, closest, final_centroids))
