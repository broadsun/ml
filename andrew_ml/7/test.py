#coding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat('data/ex7data2.mat')
X = data['X']
# print X.shape
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    #m=300, k=3, idx (1,300)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx


#idx = find_closest_centroids(X, initial_centroids)
# import pickle
# with open('tmp.pickle', 'w') as f:
#     pickle.dump(idx, f)
# with open('tmp.pickle', 'r') as f:
#     idx = pickle.load(f)


# data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
# data2.head()
# sb.set(context="notebook", style="white")
# sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
# plt.show()

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

#compute_centroids(X, idx, 3)

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return idx, centroids

#初始化聚类中心
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids
'''
initial_centroids = init_centroids(X, 3)

idx, centroids = run_k_means(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.scatter(centroids[:,0], centroids[:,1], s=30, color='k', label='centroids')
ax.legend()
plt.show()
'''



##########################################
# from IPython.display import Image
# Image(filename='data/bird_small.png')
'''
image_data = loadmat('data/bird_small.mat')

A = image_data['A']
print A.shape

### normalize value ranges
A = A / 255.

### reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print X.shape

### randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

### run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)
print idx[20:40],centroids.shape;
### get the closest centroids one last time
idx = find_closest_centroids(X, centroids)
### map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]
#print X_recovered.shape

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
# X_recovered.shape
plt.imshow(X_recovered)
plt.show()
'''
'''
from skimage import io

# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('data/bird_small.png') / 255.
# io.imshow(pic)
# plt.show()
# serialize data
data = pic.reshape(128*128, 3)
from sklearn.cluster import KMeans#导入kmeans库

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()

'''

###PCA
data = loadmat('data/ex7data1.mat')
X = data['X']

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(X[:, 0], X[:, 1])
# plt.show()

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    
    return U, S, V,cov

U, S, V, cov = pca(X)
print X.shape, U.shape, S.shape, V.shape, cov.shape

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

Z = project_data(X, U, 1)
print Z.shape
def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)
print X_recovered.shape

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
# plt.show()


faces = loadmat('data/ex7faces.mat')
X = faces['X']
print X.shape

def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
 


