import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


def run_kmeans():
    # ===== DATASET (1000 records) =====
    np.random.seed(42)

    X = np.random.randint(18, 60, 1000)
    Y = np.random.randint(1000, 5000, 1000)

    data = np.column_stack((X, Y))

    # ===== MODEL =====
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # ===== CLUSTER PLOT =====
    plt.figure(figsize=(8,6))

    plt.scatter(data[:,0], data[:,1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                label='Data Points')

    plt.scatter(centroids[:,0], centroids[:,1],
                s=250,
                c='red',
                marker='X',
                label='Centroids')

    plt.title("K-Means Clustering Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    cluster_path = os.path.join("static", "kmeans.png")
    plt.savefig(cluster_path)
    plt.close()

    # ===== VARIANCE DATA (FROM YOUR MANUAL PART) =====
    iterations = [1, 2, 3]
    variance = [528.66, 351.29, 327.55]

    # ===== VARIANCE PLOT =====
    plt.figure(figsize=(6,4))

    plt.plot(iterations, variance,
             marker='o',
             linestyle='-')

    plt.title("Variance Reduction Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")

    variance_path = os.path.join("static", "variance.png")
    plt.savefig(variance_path)
    plt.close()

    return centroids