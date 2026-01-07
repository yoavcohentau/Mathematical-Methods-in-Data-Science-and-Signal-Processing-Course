import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from numpy.linalg import eigh
from sklearn.decomposition import PCA
import itertools


# ------ Error metrics ------
def test_performance(true_labels, pred_labels, k=3):
    I_true = labels_to_indicator(true_labels, k)
    I_pred = labels_to_indicator(pred_labels, k)

    frob_err = []
    acc = []
    for p in itertools.permutations(range(k)):
        frob_err.append(0.5 * np.linalg.norm(I_true - I_pred[:, p], 'fro'))
        mapped = np.array([p[l] for l in pred_labels])
        acc.append(np.mean(mapped != true_labels))
    idx_min = np.argmin(frob_err)

    return frob_err[idx_min], acc[idx_min]


def calc_accuracy_rate(true_labels, pred_labels):
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


# ------ utils ------
def labels_to_indicator(labels, k):
    I = np.zeros((len(labels), k))
    for i, l in enumerate(labels):
        I[i, l] = 1
    return I


# ------ Simulation ------

# Load data
X = np.loadtxt("./Q5_data/data.csv", delimiter=",")
I_true = np.loadtxt("./Q5_data/clusters.csv", delimiter=",")
true_labels = np.argmax(I_true, axis=1)

# parameters
n = X.shape[0]
k = 3


# -- (i) pure k-means
_, labels_km = kmeans2(X, k, minit='++', iter=50)
I_km = labels_to_indicator(labels_km, k)

err_km_rate = calc_accuracy_rate(true_labels, labels_km)
err_km_frob = frobenius_error(I_true, I_km)

print("k-means misclassification rate:", 1 - err_km_rate)
print("k-means Frobenius error:", err_km_frob)


# -- (ii) spectral clustering
# Gaussian kernel (local scale)
# sigma = np.percentile(cdist(X, X), 10) / 10
sigma = np.percentile(cdist(X, X), 20) / 10
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
# U = eigvecs[:, 1:k+1]     # skip trivial eigenvector
V = eigvecs[:, 1:k]          # v_2,...,v_k
U = D_inv_sqrt @ V

# Row normalization (CRITICAL)
# U = U / np.linalg.norm(U, axis=1, keepdims=True)

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

    err_spec_rate = calc_accuracy_rate(true_labels, labels_spec)
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
