import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from numpy.linalg import eigh
from sklearn.decomposition import PCA
import itertools


# ------ performance metric ------
def test_performance(true_labels, pred_labels, k=3):
    I_true = labels_to_indicator(true_labels, k)
    I_pred = labels_to_indicator(pred_labels, k)

    min_frob_err = np.inf
    acc = 0.0

    # run on all permutations
    for p in itertools.permutations(range(k)):
        # reorder I
        I_pred_permuted = I_pred[:, p]

        # calc Frobenius error
        current_frob_err = 0.5 * np.linalg.norm(I_true - I_pred_permuted, 'fro')

        # calc accuracy
        pred_labels_permute = np.argmax(I_pred_permuted, axis=1)
        current_acc = np.mean(pred_labels_permute == true_labels)

        # save the optimal performance which leads to minimal Frobenius error
        if current_frob_err < min_frob_err:
            min_frob_err = current_frob_err
            acc = current_acc

    return min_frob_err, acc


# ------ utils ------
def labels_to_indicator(labels, k):
    I = np.zeros((len(labels), k))
    for i, l in enumerate(labels):
        I[i, l] = 1
    return I


# ------ k-means ------
def apply_k_means(X, k=3, n_trials=50, n_iters=50):
    min_distortion = np.inf
    labels = None
    for _ in range(n_trials):
        centroids, current_labels = kmeans2(X, k, minit='++', iter=n_iters)

        # save the min-distortion clustering
        current_distortion = np.sum((X - centroids[current_labels]) ** 2)
        if current_distortion < min_distortion:
            min_distortion = current_distortion
            labels = current_labels

    I_km = labels_to_indicator(labels, k)

    return labels, I_km


# ------ Simulation ------

# Load data
X = np.loadtxt('./Q5_data/data.csv', delimiter=',')
I_true = np.loadtxt('./Q5_data/clusters.csv', delimiter=',')
true_labels = np.argmax(I_true, axis=1)

# parameters
n = X.shape[0]
k = 3


# -- (i) pure k-means
labels_km, I_km = apply_k_means(X, k)
err_km_frob, acc_km_rate = test_performance(true_labels, labels_km)

print("k-means accuracy rate:", acc_km_rate)
print("k-means Frobenius error:", err_km_frob)


# -- (ii) spectral clustering
# Gaussian kernel (local scale)
# sigma = np.percentile(cdist(X, X), 20) / 10
sigma = 0.6
W = np.exp(-cdist(X, X)**2 / (2 * sigma**2))
np.fill_diagonal(W, 0)

# Normalized Laplacian
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
# L_sym = D - W

# Eigen-decomposition
eigvals, eigvecs = eigh(L_sym)
# U = eigvecs[:, 1:k+1]     # skip trivial eigenvector
V = eigvecs[:, 1:k]          # v_2,...,v_k
U = np.matmul(D_inv_sqrt, V)

# apply k-means on eigen-vectors
labels_spec, I_spec = apply_k_means(U, k)
err_spec_frob, acc_spec_rate = test_performance(true_labels, labels_spec)

print("Spectral accuracy rate:", acc_spec_rate)
print("Spectral Frobenius error:", err_spec_frob)


# ------ PCA visualization ------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='tab10', s=15)
plt.title("ground-truth clustering (PCA view)")
plt.axis("equal")

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='tab10', s=15)
plt.title("k-means clustering (PCA view)")
plt.axis("equal")

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_spec, cmap='tab10', s=15)
plt.title("Spectral clustering (PCA view)")
plt.axis("equal")

plt.suptitle('Data in PCA view')
plt.tight_layout()
plt.show()
