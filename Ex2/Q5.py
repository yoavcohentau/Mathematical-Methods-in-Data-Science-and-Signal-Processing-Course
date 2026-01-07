import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from numpy.linalg import eigh
from sklearn.decomposition import PCA
import itertools


# ------------------------
# Load data
# ------------------------
X = np.loadtxt("./Q5_data/data.csv", delimiter=",")
I_true = np.loadtxt("./Q5_data/clusters.csv", delimiter=",")

n = X.shape[0]
k = 3
true_labels = np.argmax(I_true, axis=1)


# ------------------------
# Error metrics
# ------------------------
def misclassification_rate(true_labels, pred_labels):
    best = np.inf
    for p in itertools.permutations(range(k)):
        mapped = np.array([p[l] for l in pred_labels])
        best = min(best, np.mean(mapped != true_labels))
    return best


def frobenius_error(I_true, I_pred):
    errs = []
    for p in itertools.permutations(range(k)):
        errs.append(0.5 * np.linalg.norm(I_true - I_pred[:, p], 'fro'))
    return min(errs)


def labels_to_indicator(labels, k):
    I = np.zeros((len(labels), k))
    for i, l in enumerate(labels):
        I[i, l] = 1
    return I


# ------------------------
# k-means
# ------------------------
_, labels_km = kmeans2(X, k, minit='++', iter=50)
I_km = labels_to_indicator(labels_km, k)

err_km_rate = misclassification_rate(true_labels, labels_km)
err_km_frob = frobenius_error(I_true, I_km)

print("k-means misclassification rate:", 1 - err_km_rate)
print("k-means Frobenius error:", err_km_frob)

# ------------------------
# Spectral clustering
# ------------------------
# Gaussian kernel (local scale)
sigma = np.percentile(cdist(X, X), 10) / 10
# sigma = 1.2561588877651955
W = np.exp(-cdist(X, X)**2 / (2 * sigma**2))
np.fill_diagonal(W, 0)

# Normalized Laplacian
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
# L_sym = D - W

# Eigen-decomposition
eigvals, eigvecs = eigh(L_sym)
U = eigvecs[:, 1:k+1]     # skip trivial eigenvector
# U = eigvecs[:, :k]

# Row normalization (CRITICAL)
U = U / np.linalg.norm(U, axis=1, keepdims=True)

# k-means on spectral embedding
# err_spec_frob = np.inf
#
# _, labels_spec = kmeans2(U, k, minit='++', iter=50)
# I_spec = labels_to_indicator(labels_spec, k)
#
# err_spec_rate = misclassification_rate(true_labels, labels_spec)
# err_spec_frob = frobenius_error(I_true, I_spec)

err_spec_rate_min = np.inf
err_spec_frob_min = np.inf
labels_spec_min = []
for _ in range(50):
    _, labels_spec = kmeans2(U, k, minit='++', iter=50)
    I_spec = labels_to_indicator(labels_spec, k)

    err_spec_rate = misclassification_rate(true_labels, labels_spec)
    err_spec_frob = frobenius_error(I_true, I_spec)

    if err_spec_frob < err_spec_frob_min:
        labels_spec_min = labels_spec
        err_spec_rate_min = err_spec_rate
        err_spec_frob_min = err_spec_frob


print("Spectral misclassification rate:", 1 - err_spec_rate)
print("Spectral Frobenius error:", err_spec_frob)




# ------------------------
# PCA visualization
# ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='tab10', s=15)
plt.title("k-means clustering (PCA view)")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_spec, cmap='tab10', s=15)
plt.title("Spectral clustering (PCA view)")
plt.axis("equal")

plt.tight_layout()
plt.show()



# import numpy as np
# from scipy.cluster.vq import kmeans2
# from scipy.spatial.distance import cdist
# from numpy.linalg import eigh
# import itertools
#
#
# DATA_FILE_PATH = rf'./Q5_data/data.csv'
# CLUSTER_FILE_PATH = rf'./Q5_data/clusters.csv'
#
# X = np.loadtxt(DATA_FILE_PATH, delimiter=",")
# I_true = np.loadtxt(CLUSTER_FILE_PATH, delimiter=",")
#
#
# n = X.shape[0]
# k = 3
#
#
# def clustering_error(I_true, I_pred):
#     errs = []
#     errs_counter = []
#     for p in itertools.permutations(range(3)):
#         errs.append(0.5 * np.linalg.norm(I_true - I_pred[:, p], 'fro'))
#         errs_counter.append(0.5*np.count_nonzero(I_true - I_pred[:, p]))
#     return min(errs)
#
#
# # 5-(i)
# centroids, labels = kmeans2(
#     X,
#     3,
#     minit='++',
#     iter=50
# )
#
# I_kmeans = np.zeros((n, 3))
# for i in range(n):
#     I_kmeans[i, labels[i]] = 1
#
# err_kmeans = clustering_error(I_true, I_kmeans)
# print("k-means error:", err_kmeans)
#
#
# # 5-(i)
# sigma = np.median(cdist(X, X))
# W = np.exp(-cdist(X, X)**2 / (2*sigma**2))
# np.fill_diagonal(W, 0)
#
#
# D = np.diag(W.sum(axis=1))
# L = D - W
#
#
# eigvals, eigvecs = eigh(L)
# U = eigvecs[:, :3]
#
#
# centroids_spec, labels_spec = kmeans2(
#     U,
#     3,
#     minit='++',
#     iter=50
# )
#
# I_spec = np.zeros((n, 3))
# for i in range(n):
#     I_spec[i, labels_spec[i]] = 1
#
# err_spec = clustering_error(I_true, I_spec)
# print("Spectral clustering error:", err_spec)
#
#
#
#
# # import numpy as np
# # from sklearn.cluster import KMeans
# # from scipy.spatial.distance import cdist
# # from numpy.linalg import eigh
# # import itertools
# #
# #
# # DATA_FILE_PATH = rf'./Q5_data/data.csv'
# # CLUSTER_FILE_PATH = rf'./Q5_data/clusters.csv'
# #
# # X = np.loadtxt(DATA_FILE_PATH, delimiter=",")
# # I_true = np.loadtxt(CLUSTER_FILE_PATH, delimiter=",")
# #
# #
# # n = X.shape[0]
# # k = 3
# #
# #
# # def clustering_error(I_true, I_pred):
# #     errs = []
# #     for p in itertools.permutations(range(3)):
# #         errs.append(0.5 * np.linalg.norm(I_true - I_pred[:, p], 'fro'))
# #     return min(errs)
# #
# #
# # # 5-(i)
# # kmeans = KMeans(n_clusters=3, n_init=20)
# # labels = kmeans.fit_predict(X)
# #
# # I_kmeans = np.zeros((n, 3))
# # for i in range(n):
# #     I_kmeans[i, labels[i]] = 1
# #
# # err_kmeans = clustering_error(I_true, I_kmeans)
# # print("k-means error:", err_kmeans)
# #
# #
# # # 5-(i)
# # sigma = np.median(cdist(X, X))
# # W = np.exp(-cdist(X, X)**2 / (2*sigma**2))
# # np.fill_diagonal(W, 0)
# #
# #
# # D = np.diag(W.sum(axis=1))
# # L = D - W
# #
# #
# # eigvals, eigvecs = eigh(L)
# # U = eigvecs[:, :3]
# #
# #
# # kmeans = KMeans(n_clusters=3, n_init=20)
# # labels_spec = kmeans.fit_predict(U)
# #
# # I_spec = np.zeros((n, 3))
# # for i in range(n):
# #     I_spec[i, labels_spec[i]] = 1
# #
# # err_spec = clustering_error(I_true, I_spec)
# # print("Spectral clustering error:", err_spec)
# #
# #
