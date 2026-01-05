import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


# parameters
SIGNAL_LENGTH = 50
WINDOW_SIZE = 10
N = 2000

# base signal x
x = np.zeros(SIGNAL_LENGTH)
x[:WINDOW_SIZE] = 1


def circular_shift(x, shift):
    return np.roll(x, shift)


def generate_data(sigma, num_of_samples=N):
    shifts = np.random.randint(0, SIGNAL_LENGTH, size=num_of_samples)
    y = np.array([circular_shift(x, l) for l in shifts])
    noise = sigma * np.random.randn(num_of_samples, SIGNAL_LENGTH)
    return y + noise, shifts


def diffusion_map(y, tau, n_components=3):
    # square distances matrix
    dist2 = cdist(y, y, metric='sqeuclidean')

    # find M using gaussian kernel
    W = np.exp(-dist2 / tau**2)
    D = np.sum(W, axis=1)
    M = W / D[:, None]

    # eigen-decomposition
    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals[:n_components], eigvecs[:, :n_components]


# ------ a - (i) ------
sigma = 0
y_clean, shifts = generate_data(sigma=0)

# choose tau_g close to the mean of distances between vectors
tau_g = int(np.ceil(np.median(cdist(y_clean, y_clean))))

eigvals, eigvecs = diffusion_map(y_clean, tau_g)

plt.figure()
plt.plot(eigvecs[:, 2])
plt.show()

# ------ a - (ii) ------
tau_g_small = tau_g / 10
eigvals_small, eigvecs_small = diffusion_map(y_clean, tau_g_small)


# ------ a - plot results ------
plt.figure(figsize=(8, 4))

# Left plot: tau_g
plt.subplot(1, 2, 1)
plt.scatter(eigvecs[:, 1], eigvecs[:, 2])
plt.title(rf'$\tau_g = {tau_g}$')
plt.axis('equal')

# Right plot: tau_g / 10
plt.subplot(1, 2, 2)
plt.scatter(eigvecs_small[:, 1], eigvecs_small[:, 2])
plt.title(rf'$\tau_g = {tau_g_small}$')
plt.axis('equal')

plt.suptitle('Diffusion Map Embedding - Without Noise')
plt.show()


# ------ b ------
sigma_vec = [0.01, 0.1, 0.5, 1]

plt.figure(figsize=(6, 6))

for sigma_idx in range(len(sigma_vec)):
    sigma = sigma_vec[sigma_idx]
    y_noisy, shifts = generate_data(sigma)

    eigvals, eigvecs = diffusion_map(y_noisy, tau_g)

    plt.subplot(2, 2, sigma_idx+1)
    plt.scatter(eigvecs[:, 1], eigvecs[:, 2])
    plt.title(rf'$\sigma = {sigma}$')
    plt.axis('equal')

plt.suptitle('Diffusion Map Embedding - With Noise')
plt.show()
