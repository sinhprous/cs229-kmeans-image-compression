import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

vectors = scipy.misc.imread("mandrill-small.png").astype(np.float).reshape((-1, 3))

# distance between 2 vectors
def distance (v, w, dim):
    return np.sqrt(np.sum([(v[i]-w[i])**2 for i in range(dim)]))

# kmeans
def kmeans(vectors, dim, k, max_iters = 30, threshold = 0.01):
    centroidIndexs = np.random.choice(range(len(vectors)), k, replace=False)
    centroids = vectors[centroidIndexs]
    cluster = [0.0]*len(vectors)
    for iter in range(max_iters):
        converge = True
        print("*")
        for j in range(len(vectors)):
            cluster[j] = np.argmin([distance(vectors[j], centroids[i], dim) for i in range(k)])
        new_centroids = [[0.0]*dim]*k
        for i in range(k):
            new_centroids[i] = np.mean([vectors[j] for j in range(len(vectors)) if cluster[j] == i], axis = 0)
            if distance(new_centroids[i], centroids[i], dim) < threshold:
                converge = converge * True
            else:
                converge = converge * False
                centroids[i] = new_centroids[i]
        if converge == True:
            break
    return centroids, cluster

num_cluster = 16
centroids, cluster = kmeans(vectors, vectors.shape[1], num_cluster, 30, 0.01)
compressedImage = np.array([centroids[i] for i in cluster])
plt.imshow(np.uint8(compressedImage.reshape(np.sqrt(compressedImage.shape[0]),-1,3)))
plt.show()