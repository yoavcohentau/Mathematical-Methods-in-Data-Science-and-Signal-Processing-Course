import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def generate_graph(p, n=40):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            same_group_indicator = (i < 20 and j < 20) or (i >= 20 and j >= 20)
            prob = p if same_group_indicator else 1 - p
            if np.random.rand() < prob:
                A[i, j] = A[j, i] = 1
    return A


def maxcut_sdp(A):
    n = A.shape[0]
    X = cp.Variable((n, n), PSD=True)  # optimization matrix variable definition
    objective = cp.Maximize(cp.sum(cp.multiply(A, 1 - X)))  # the maximization problem
    constraints = [cp.diag(X) == 1]  # constraints - 1's on the diagonal
    prob = cp.Problem(objective, constraints)  # optimization object
    prob.solve(solver=cp.SCS, verbose=False)  # solve the optimization problem
    return X.value


def clustering_error(X):
    # find the labels from the relaxation solution matrix X
    eigvals, eigvecs = np.linalg.eigh(X)
    v = eigvecs[:, -1]
    labels = np.sign(v)

    # create the true indicators vector
    true_indicators_vector = np.ones(40)
    true_indicators_vector[20:] = -1

    # the solution can be the negative representation
    err1 = np.mean(labels != true_indicators_vector)
    err2 = np.mean(-labels != true_indicators_vector)
    return min(err1, err2)


# Adjacency matrix examples
A_p01 = generate_graph(p=0.1)
A_p05 = generate_graph(p=0.5)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(A_p01, cmap='plasma')
plt.title('p = 0.1')

plt.subplot(1, 2, 2)
plt.imshow(A_p05, cmap='plasma')
plt.title('p = 0.5')

plt.suptitle('Adjacency Matrix', fontsize=14)
plt.show()


# Complex relaxation of max-cut simulation
p_vec = np.linspace(0.1, 0.5, 20)
errors = []

for p in p_vec:
    err = []
    for _ in range(50):
        A = generate_graph(p)
        X = maxcut_sdp(A)
        err.append(clustering_error(X))
    errors.append(np.mean(err))

plt.plot(p_vec, errors, marker='o')
plt.xlabel('p')
plt.ylabel('Average clustering error')
plt.title('Convex relaxation of Max-Cut')
plt.grid(True)
plt.show()
