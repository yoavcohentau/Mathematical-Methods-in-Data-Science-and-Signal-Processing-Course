import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# parameters
n = 1000
T = 2000
first_idx_linear_line_estim = int(T / 5)

# random symmetric matrix
A = np.random.randn(n, n)  # random A coefficients
A = (A + A.T) / 2          # symmetrize A

# ground truth: dominant eigenpairs (largest magnitude - 'LM')
eigvals, eigvecs = eigsh(A, k=2, which='LM')
sorted_idx = np.argsort(np.abs(eigvals))[::-1]
lambda1, lambda2 = eigvals[sorted_idx[0]], eigvals[sorted_idx[1]]
v = eigvecs[:, sorted_idx[0]]
v = v / np.linalg.norm(v)

# power iteration
u = np.random.randn(n)
u = u / np.linalg.norm(u)
errors = []

for i in range(T):
    u = np.matmul(A, u)
    u = u / np.linalg.norm(u)
    err = min(
        np.linalg.norm(u - v),
        np.linalg.norm(u + v)
    )
    errors.append(err)

errors = np.array(errors)
iters = np.arange(T)


# ---------- empirical linear fit ----------
log_err = np.log(errors[first_idx_linear_line_estim:])
fit_iters = iters[first_idx_linear_line_estim:]
slope_emp, intercept_emp = np.polyfit(fit_iters, log_err, 1)
fit_line = np.exp(intercept_emp + slope_emp * iters)

# ---------- theoretical line ----------
theoretical_slope = np.log(abs(lambda2 / lambda1))

# choose intercept so that the line matches error at `first_idx_linear_line_estim`
theoretical_intercept = np.log(errors[first_idx_linear_line_estim]) - theoretical_slope * first_idx_linear_line_estim
theoretical_line = np.exp(theoretical_intercept + theoretical_slope * iters)

# ---------- plot ----------
plt.semilogy(errors, label='Power Iteration error')
plt.semilogy(fit_line, '--', label='Empirical linear fit')
plt.semilogy(theoretical_line, ':', label='Theoretical slope')
plt.xlabel('Iteration')
plt.ylabel('Relative error')
plt.title('Power Iteration Convergence')
plt.legend()
plt.grid(True)
plt.show()

# ---------- print slopes ----------
print(f"Empirical slope:   {slope_emp:.4e}")
print(f"Theoretical slope: {theoretical_slope:.4e}")
