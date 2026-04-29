import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------
# GÉNÉRATION DE DONNÉES
# -----------------------------
np.random.seed(0)

cluster_1 = np.random.randn(100, 2) + [2, 2]
cluster_2 = np.random.randn(100, 2) + [-2, -2]
cluster_3 = np.random.randn(100, 2) + [2, -2]

X = np.vstack((cluster_1, cluster_2, cluster_3))

# -----------------------------
# K-MEANS
# -----------------------------
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# -----------------------------
# VISUALISATION
# -----------------------------
plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

plt.title("Clustering avec K-Means")
plt.xlabel("X")
plt.ylabel("Y")

plt.grid()
plt.show()
