import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import sys

matrix = np.loadtxt(sys.argv[1], delimiter=",")
N = len(matrix)
matrix = 1.0 - matrix

model = AgglomerativeClustering(n_clusters=None, linkage="average", affinity="precomputed", distance_threshold=0.2)
model.fit(matrix)

cluster_sizes = {}
for c in model.labels_:
    if c in cluster_sizes:
        cluster_sizes[c] += 1
    else:
        cluster_sizes[c] = 1

remap = dict([(j, i) for i, (j, sz) in enumerate(sorted(cluster_sizes.items(), key=lambda item: item[1] if item[0] != -1 else -1e10))])
mapped = [remap[i] for i in model.labels_]
reordering = (list(zip(*sorted(list(zip(mapped, range(N))),key=lambda item: -item[0] if item[0] != -1 else 1e10)))[1])

for i in range(N):
    for j in range(N):
        matrix[i][j] = 1.0 - matrix[i][j]

plt.matshow(matrix[np.ix_(reordering, reordering)])
plt.show()

# print(dict(enumerate(reordering)))
