import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000
p = 10

# Random vectors x1, x2
np.random.seed(0)
x1 = np.random.randn(p)
x2 = np.random.randn(p)

# S values (logarithmic scale)
S_values = np.logspace(0, 6, 30)  # from 1 to 10^6

cov_eig_val_ratios = []
sing_val_ratios = []

for S in S_values:
    # Generate coefficients
    a = np.random.randn(n)
    b = np.random.randn(n)

    # Generate data matrix Y size pXn
    Y = S * np.outer(x1, a) + np.outer(x2, b)

    # --- (a) Covariance eigenvalues ---
    C = (1/n) * np.matmul(Y, Y.T)
    cov_eig_vals = np.linalg.eigvalsh(C)  # sorted ascending
    cov_eig_ratio = cov_eig_vals[-1] / cov_eig_vals[-2]
    cov_eig_val_ratios.append(cov_eig_ratio)

    # --- (b) Singular values ---
    s = np.linalg.svd(Y, compute_uv=False)  # sorted descending
    sing_ratio = s[0] / s[1]
    sing_val_ratios.append(sing_ratio)

# Convert to np.array
cov_eig_val_ratios = np.array(cov_eig_val_ratios)
sing_val_ratios = np.array(sing_val_ratios)

# Fit slopes in log-log using linear regression
S_values_log = np.log(S_values)
cov_eig_val_ratios_log = np.log(cov_eig_val_ratios)
sing_val_ratios_log = np.log(sing_val_ratios)

slope_eig = np.polyfit(S_values_log, cov_eig_val_ratios_log, 1)[0]
slope_sing = np.polyfit(S_values_log, sing_val_ratios_log, 1)[0]

print("Eigenvalue ratio slope (expected ≈ 2):", slope_eig)
print("Singular value ratio slope (expected ≈ 1):", slope_sing)

# Plot
plt.figure()
plt.loglog(S_values, cov_eig_val_ratios, 'o-', label=f"Eigenvalues ratio, slope={slope_eig:.3f}")
plt.loglog(S_values, sing_val_ratios, 'o-', label=f"Singular values ratio, slope={slope_sing:.3f}")
plt.xlabel('S')
plt.ylabel('Ratio')
plt.grid(True, which='both')
plt.legend()
plt.title("Eigenvalue & Singular Value Ratios")
plt.show()
