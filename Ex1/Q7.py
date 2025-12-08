import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
n = 2000
p = 500
gamma = p / n

# (a) - normal distribution model
X = np.random.randn(p, n)  # pXn
cov = (1.0 / n) * np.matmul(X, X.T)  # pXp
eig_vals = np.linalg.eigvalsh(cov)

# Marchenko-Pastur distibution
lambda_minus = (1.0 - np.sqrt(gamma))**2
lambda_plus = (1.0 + np.sqrt(gamma))**2


def marcenko_pastur_distribution(lam, gamma):
    lam_minus = (1.0 - np.sqrt(gamma))**2
    lam_plus = (1.0 + np.sqrt(gamma))**2
    dist = np.zeros_like(lam)
    mask = (lam > lam_minus) & (lam < lam_plus)
    dist[mask] = np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus))\
                 / (2.0 * np.pi * gamma * lam[mask])
    return dist


# calc eigen values histogram and Marcenko-Pastur distribution
xs = np.linspace(max(1e-8, lambda_minus*0.99), lambda_plus*1.01, 1000)
theoretical_distribution = marcenko_pastur_distribution(xs, gamma)

# Plot
plt.figure()
vals, bins, _ = plt.hist(eig_vals, bins=40, density=True, label='Empirical eigenvalues distribution')
plt.plot(xs, theoretical_distribution, label='Marchenko–Pastur distribution')
plt.xlabel('Eigenvalue')
plt.ylabel('Distribution')
plt.title(f'Normal Model\nEmpirical eigenvalues vs. Marchenko–Pastur\nn={n}, p={p}, gamma={gamma:.3f}')
plt.legend()
# plt.show()


# Part (b) - Spiked model (rank-1 perturbation)
def simulate_spiked_cov(beta):
    """
    the model: Y = g + sqrt(beta)*g0*u
    """
    u = np.random.randn(p)
    u /= np.linalg.norm(u)

    g = np.random.randn(p, n)
    spike = np.sqrt(beta) * u[:, None] * np.random.randn(1, n)
    Y = g + spike

    cov_b = np.matmul(Y, Y.T) / n  # covariance
    return np.linalg.eigvalsh(cov_b)


beta_critic = np.sqrt(gamma)
print(f"beta critic = {beta_critic}")

beta_small = 0.3  # before beta_critic
beta_large = 1.0  # after beta_critic

# Compute eigenvalues
eig_small = simulate_spiked_cov(beta_small)
eig_large = simulate_spiked_cov(beta_large)


# ---- Plot 1: before beta_critic ----
plt.figure()
plt.hist(eig_small, bins=40, density=True, label='Empirical eigenvalues distribution')
plt.plot(xs, theoretical_distribution, label='Marchenko–Pastur distribution')
plt.xlabel('Eigenvalue')
plt.ylabel('Distribution')
plt.title(f'Spike Model\nEmpirical eigenvalues vs. Marchenko–Pastur\nn={n}, p={p}, gamma={gamma:.3f}\nbeta < beta_critic')
plt.legend()
# plt.show()

# ---- Plot 2: after beta_critic ----
plt.figure()
plt.hist(eig_large, bins=40, density=True, label='Empirical eigenvalues distribution')
plt.plot(xs, theoretical_distribution, label='Marchenko–Pastur distribution')
plt.xlabel('Eigenvalue')
plt.ylabel('Distribution')
plt.title(f'Spike Model\nEmpirical eigenvalues vs. Marchenko–Pastur\nn={n}, p={p}, gamma={gamma:.3f}\nbeta > beta_critic')
plt.legend()

plt.show()
