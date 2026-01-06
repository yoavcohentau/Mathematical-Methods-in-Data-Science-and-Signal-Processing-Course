import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import norm


def generate_data(n, p):
    theta = np.random.uniform(0, 2*np.pi, n)
    h = np.exp(1j * theta)
    H = np.outer(h, np.conj(h))

    for j in range(n):
        for k in range(j+1, n):
            if np.random.rand() < p:
                alpha = np.random.uniform(0, 2*np.pi)
                H[j, k] = np.exp(1j * alpha)
                H[k, j] = np.exp(-1j * alpha)

    return h, H


def spectral_sync(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    v = eigvecs[:, -1]
    return v / np.abs(v)


# def sdp_sync(H):
#     n = H.shape[0]
#     X = cp.Variable((n, n), complex=True, PSD=True)
#     objective = cp.Maximize(cp.real(cp.trace(H @ X)))
#     constraints = [cp.diag(X) == 1]
#     prob = cp.Problem(objective, constraints)
#     prob.solve(solver=cp.SCS, verbose=False)
#
#     eigvals, eigvecs = np.linalg.eigh(X.value)
#     v = eigvecs[:, -1]
#     return v / np.abs(v)

# def sdp_sync(H):
#     n = H.shape[0]
#
#     Hr = np.real(H)
#     Hi = np.imag(H)
#
#     X = cp.Variable((2*n, 2*n), PSD=True)
#
#     # objective: trace(HX)
#     objective = cp.Maximize(
#         cp.trace(Hr @ X[:n, :n] - Hi @ X[:n, n:]
#                + Hi @ X[n:, :n] + Hr @ X[n:, n:])
#     )
#
#     constraints = []
#     for i in range(n):
#         constraints.append(X[i, i] + X[i+n, i+n] == 1)
#
#     prob = cp.Problem(objective, constraints)
#     prob.solve(solver=cp.SCS, verbose=False)
#
#     # extract top eigenvector
#     eigvals, eigvecs = np.linalg.eigh(X.value)
#     v = eigvecs[:, -1]
#
#     h_hat = v[:n] + 1j * v[n:]
#     return h_hat / np.abs(h_hat)
# def sdp_sync(H):
#     n = H.shape[0]
#
#     # X real symmetric
#     X = cp.Variable((n, n), symmetric=True)
#
#     objective = cp.Maximize(cp.trace(np.real(H) @ X))
#     constraints = [X >> 0, cp.diag(X) == 1]
#
#     prob = cp.Problem(objective, constraints)
#     prob.solve(solver=cp.SCS, verbose=False)
#
#     # rank-1 extraction
#     eigvals, eigvecs = np.linalg.eigh(X.value)
#     v = eigvecs[:, -1]
#
#     # lift to complex phases
#     h_hat = np.exp(1j * np.angle(v))
#     return h_hat
def sdp_sync(H):
    n = H.shape[0]

    X = cp.Variable((n, n), hermitian=True)

    objective = cp.Maximize(
        cp.real(cp.sum(cp.multiply(np.conj(H), X)))
    )

    constraints = [
        X >> 0,
        cp.diag(X) == 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    eigvals, eigvecs = np.linalg.eigh(X.value)
    v = eigvecs[:, -1]

    return v / np.abs(v)


def sync_error(h, h_hat):
    theta_al = np.vdot(h_hat, h)
    theta_al = theta_al / np.abs(theta_al)
    # return np.mean(np.abs(h_hat * theta_al - h)**2)
    return np.mean(np.abs(h - h_hat * theta_al)**2)
    # return norm(h_hat * theta_al - h, ord=2)
    # return norm(h - h_hat * theta_al, ord=2)
# # def sync_error(h, h_hat):
# #     alpha = np.vdot(h_hat, h)
# #     alpha = alpha / np.abs(alpha)
# #     return np.mean(np.abs(h - alpha * h_hat)**2)



n = 50
ps = np.linspace(0, 0.5, 11)

err_spec = []
err_sdp = []

n_experiments_per_p = 5
for p in ps:
    e1, e2 = [], []
    for _ in range(n_experiments_per_p):
        h, H = generate_data(n, p)
        h_spec = spectral_sync(H)
        h_sdp = sdp_sync(H)

        e1.append(sync_error(h, h_spec))
        e2.append(sync_error(h, h_sdp))

    err_spec.append(np.mean(e1))
    err_sdp.append(np.mean(e2))
    print(f"p={p}:  spectral error = {err_spec[-1]},  sdp error = {err_sdp[-1]}")

plt.plot(ps, err_spec, '-o', label="Spectral")
plt.plot(ps, err_sdp, '-s', label="SDP")
plt.xlabel("Outlier probability p")
plt.ylabel("Average error")
plt.legend()
plt.grid(True)
plt.show()
