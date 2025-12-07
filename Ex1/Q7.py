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








#
# # population covariance = I + beta * u u^T  => population top eigenvalue = 1 + beta
# # BBP threshold: theta = 1 + beta; separates if theta > 1 + sqrt(gamma)  -> beta > sqrt(gamma)
#
# # prepare spike model
# u = np.random.randn(p)
# u /= np.linalg.norm(u)
#
# # range of beta to test (includes below and above threshold)
# beta_values = np.concatenate((np.linspace(0.0, 0.8, 17), np.linspace(1.0, 6.0, 12)))
# top_eigs = []
#
# for beta in beta_values:
#     # the model: Y = g + sqrt(beta)*g0*u
#     g = np.random.randn(p, n)
#     g0 = np.random.randn(n)
#     Y = g + np.sqrt(beta) * np.outer(u, g0)
#     Sn = (1.0 / n) * np.matmul(Y, Y.T)
#     eig_vals_b = np.linalg.eigvalsh(Sn)
#     top_eigs.append(eig_vals_b[-1])
#
# top_eigs = np.array(top_eigs)
#
#
# # Theoretical eigenvalues - slide 16
# def theoretical_spike_sample_eig(beta, gamma):
#     lambda_plus = (np.sqrt(gamma) + 1) ** 2
#     thresh = np.sqrt(gamma)
#     if beta <= thresh:
#         return lambda_plus
#     else:
#         return (beta + 1) * (1 + gamma / beta)
#
#
# theoretical_eig_vals = np.array([theoretical_spike_sample_eig(b, gamma) for b in beta_values])
#
#
# # Plot empirical top eigenvalue vs beta with theoretical curve and bulk edge
# plt.figure(figsize=(8,5))
# plt.plot(beta_values, top_eigs, 'o-', label='Empirical top eigenvalue')
# plt.plot(beta_values, theoretical_eig_vals, 'r--', label='Theoretical BBP sample eigenvalue')
# plt.axhline(lambda_plus, color='k', linestyle=':', label=f'Bulk edge λ+={lambda_plus:.3f}')
# plt.axvline(np.sqrt(gamma), color='gray', linestyle='--', label=f'β_c = √γ = {np.sqrt(gamma):.3f}')
# plt.xlabel('Spike strength β')
# plt.ylabel('Top empirical eigenvalue')
# plt.title(f'Spiked model: top eigenvalue vs β (n={n}, p={p}, γ={gamma:.3f})')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # Show two example histograms (below and above threshold)
# def show_hist_for_beta(beta, title_suffix=''):
#     a = np.random.randn(n)
#     W = np.random.randn(p, n)
#     Y = np.sqrt(beta) * np.outer(u, a) + W
#     Cb = (1.0 / n) * (Y @ Y.T)
#     eigs_b = np.linalg.eigvalsh(Cb)
#     plt.figure(figsize=(7,4))
#     plt.hist(eigs_b, bins=80, density=True, alpha=0.6)
#     xs = np.linspace(max(1e-8, lambda_minus*0.99), lambda_plus*1.01, 1000)
#     plt.plot(xs, marcenko_pastur_distribution(xs, gamma), 'r-', lw=2)
#     plt.axvline(lambda_plus, color='k', linestyle=':', label=f'λ+={lambda_plus:.3f}')
#     plt.title(f'Eigenvalue histogram for β={beta} {title_suffix}')
#     plt.xlabel('Eigenvalue')
#     plt.ylabel('Distribution')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# show_hist_for_beta(0.2, '(below threshold)')
# show_hist_for_beta(2.0, '(above threshold)')
#
#
# # Find minimal beta where empirical top eigenvalue exceeds bulk edge by a margin
# margin = 0.05 * lambda_plus   # adjustable margin (5% of lambda_plus)
# detected_indices = np.where(top_eigs > lambda_plus + margin)[0]
# if detected_indices.size > 0:
#     beta_detected = beta_values[detected_indices[0]]
#     print(f"Detected spike popping out at empirical beta ≈ {beta_detected:.3f} (first beta with top_eig > λ+ + margin)")
# else:
#     print("No spike detected in the tested beta range (increase range or lower margin).")
#
# print(f"Theoretical BBP threshold β_c = sqrt(gamma) = {np.sqrt(gamma):.3f}")
