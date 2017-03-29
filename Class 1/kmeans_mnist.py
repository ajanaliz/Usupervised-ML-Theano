# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import plot_k_means, get_simple_data
from datetime import datetime


def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


# this function only requires the cluster assignments, which are the responsibilities, which is the labels.
def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
    for k in range(K):  # loop through all the clusters
        best_target = -1  # we don't strictly need to store this
        max_intersection = 0
        for j in range(K):  # loop through all the target labels
            """we get the intersection by looking at the R matrix. Remember that the shape of R is NxK.
            we want only the rows which correspond to this target label, which is j. That's what the first
            index is. The second index is which cluster K we're currently looking at. Finally, we take the
            sum of all these responsibilities since that's how much that data point belongs to this cluster.
            Note that in the case of hard K-means, where R only contains the values 0 or 1, this equation is
            still valid. --> we find the best j corresponding to the best intersection and add that to the
            final purity."""
            intersection = R[Y == j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N  # the last step is to divide by N so that it's independent of the number of data points.


# calculate the Davies-Bouldin Index. For this we need all the data points X, the means M, and the responsibilities R.
"""notice that even though this equation involves K, remember that lower is better, meaning that if K is
very high we'll get a better score.
so this still doesnt save us from reaching the trivial case of where K=N and every data point is its own
cluster."""


def DBI(X, M, R):
    # lower is better
    # N, D = X.shape
    # _, K = R.shape
    K, D = M.shape

    # get sigmas first
    sigma = np.zeros(K)
    for k in range(
            K):  # this loop is for calculating the sigmas which are the average distance between all the datapoints in the cluster k from the center, but since every point could potentially be part of this cluster, we need to use all of X. We can then weight it by R[k] later.
        diffs = X - M[k]  # should be NxD
        # assert(len(diffs.shape) == 2 and diffs.shape[1] == D)
        squared_distances = (diffs * diffs).sum(axis=1)
        # assert(len(squared_distances.shape) == 1 and len(squared_distances) != D)
        weighted_squared_distances = R[:, k] * squared_distances
        sigma[k] = np.sqrt(weighted_squared_distances).mean()

    # calculate Davies-Bouldin Index using the sigmas we just calculated
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(
                    M[k] - M[j])  # denominator = distance between the cluster center K and the cluster center J
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K  # divide the sum of all the max ratios by K.


def main():
    X, Y = get_data(1000)

    # simple data
    # X = get_simple_data()
    # Y = np.array([0]*300 + [1]*300 + [2]*300)

    print("Number of data points:", len(Y))
    # Note: I modified plot_k_means from the original
    # lecture to return means and responsibilities
    # print "performing k-means..."
    # t0 = datetime.now()
    M, R = plot_k_means(X, len(set(Y)))
    # print "k-means elapsed time:", (datetime.now() - t0)
    # Exercise: Try different values of K and compare the evaluation metrics
    print("Purity:", purity(Y, R))
    print("DBI:", DBI(X, M, R))


if __name__ == "__main__":
    main()
