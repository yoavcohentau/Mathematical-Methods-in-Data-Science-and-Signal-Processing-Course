import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def generate_data(n, p):
    theta = np.random.uniform(0, 2*np.pi, n)
    h = np.exp(1j * theta)
    H = np.outer(h, np.conj(h))  # create H matrix

    # add outliers in prob p
    for j in range(n):
        for k in range(j+1, n):
            if np.random.rand() < p:
                alpha = np.random.uniform(0, 2*np.pi)
                H[j, k] = np.exp(1j * alpha)
                H[k, j] = np.exp(-1j * alpha)  # save H hermitian

    return h, H


def spectral_sync(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    v = eigvecs[:, -1]  # take the largest eigenvector
    return v / np.abs(v)


def sdp_sync(H):
    n = H.shape[0]
    X = cp.Variable((n, n), hermitian=True)  # optimization matrix variable definition
    objective = cp.Maximize(cp.real(cp.sum(cp.multiply(np.conj(H), X))))  # the maximization problem
    constraints = [
        X >> 0,
        cp.diag(X) == 1  # 1's on the diagonal
    ]
    prob = cp.Problem(objective, constraints)  # optimization object
    prob.solve(solver=cp.SCS, verbose=False)  # solve the optimization problem

    eigvals, eigvecs = np.linalg.eigh(X.value)
    v = eigvecs[:, -1]  # take the largest eigenvector

    return v / np.abs(v)


def sync_error(h, h_hat):
    # best alignment
    theta_al = np.vdot(h_hat, h)
    theta_al = theta_al / np.abs(theta_al)
    return np.mean(np.abs(h - h_hat * theta_al)**2)  # return error value


# ------ Simulation ------

n = 100
p_vec_size = 11
p_vec = np.linspace(0, 0.5, p_vec_size)
n_experiments_per_p = 50

err_spectral = []
err_sdp = []

for p in p_vec:
    e1, e2 = [], []
    for _ in range(n_experiments_per_p):
        h, H = generate_data(n, p)
        h_spectral = spectral_sync(H)
        h_sdp = sdp_sync(H)

        e1.append(sync_error(h, h_spectral))
        e2.append(sync_error(h, h_sdp))

    err_spectral.append(np.mean(e1))
    err_sdp.append(np.mean(e2))
    print(f"p={p:.2f}:  spectral error = {err_spectral[-1]},  sdp error = {err_sdp[-1]}")

# Plot
plt.plot(p_vec, err_spectral, '-o', label='Spectral')
plt.plot(p_vec, err_sdp, '-s', label='SDP')
plt.xlabel('Outlier probability p')
plt.ylabel('Mean error')
plt.title('Spectral vs. SDP [error graphs]')
plt.legend()
plt.grid(True)
plt.show()
