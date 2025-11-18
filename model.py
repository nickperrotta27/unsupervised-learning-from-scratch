from google.colab import files

uploaded = files.upload()
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import silhouette_score

# Load data
with open("kmeans_data.pkl", "rb") as f:
    X = pickle.load(f)

def initialize_centroids(X, K, random_state=42):
    """Initialize K centroids using K-Means++."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    centroids = np.empty((K, d))
    # Pick first centroid randomly
    centroids[0] = X[rng.integers(n)]
    # Initialize distances
    dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for k in range(1, K):
        probs = dist_sq / dist_sq.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
        new_dist_sq = np.sum((X - centroids[k]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return centroids


def assign_clusters(X, centroids):
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, K):
    """Update centroids as the mean of assigned points."""
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        members = X[labels == k]
        if len(members) > 0:
            centroids[k] = members.mean(axis=0)
        else:
            # Reinitialize empty cluster randomly
            centroids[k] = X[np.random.randint(0, len(X))]
    return centroids


def within_cluster_variance(X, labels, centroids):
    """Compute average within-cluster variance."""
    return np.mean(np.sum((X - centroids[labels]) ** 2, axis=1))


def kmeans(X, K, max_iter=200, tol=1e-6, random_state=42, verbose=False):
    centroids = initialize_centroids(X, K, random_state)
    prev_centroids = centroids.copy()
    variance_history = []

    for i in range(max_iter):
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, K)
        var = within_cluster_variance(X, labels, centroids)
        variance_history.append(var)

        shift = np.linalg.norm(centroids - prev_centroids)
        if verbose:
            print(f"Iteration {i+1}: variance={var:.6f}, centroid shift={shift:.6e}")
        if shift < tol:
            break
        prev_centroids = centroids.copy()

    return centroids, labels, variance_history


# -------------------------------------------------------
# Run K-Means for a chosen K
# -------------------------------------------------------
K = 3  # Example initial choice (you can change)
centroids, labels, var_history = kmeans(X, K, random_state=42, verbose=True)

# Plot convergence (variance vs iteration)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(var_history) + 1), var_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Average Within-Cluster Variance")
plt.title(f"K-Means Convergence (K={K})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=100, marker="X", label="Centroids")
plt.legend()
plt.title(f"K-Means Clustering (K={K})")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Elbow Method
# -------------------------------------------------------
K_range = range(1, 10)
sse = []
for k in K_range:
    centroids, labels, _ = kmeans(X, k, random_state=42)
    sse_val = np.sum((X - centroids[labels]) ** 2)
    sse.append(sse_val)

plt.figure(figsize=(6, 4))
plt.plot(K_range, sse, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.tight_layout()
plt.show()
# -------------------------------------------------------
# Silhouette Method
# -------------------------------------------------------
silhouette_scores = []
for k in range(2, 10):
    centroids, labels, _ = kmeans(X, k, random_state=42)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 4))
plt.plot(range(2, 10), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal K")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Find best K from Silhouette ---
best_k_silhouette = range(2, 10)[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
print(f"Best Silhouette K = {best_k_silhouette} (score = {best_score:.3f})")

# --- Find best K from Elbow (optional heuristic) ---
# We'll pick the K where the drop in SSE slows down significantly
sse_diffs = np.diff(sse)
elbow_k = K_range[np.argmin(np.abs(np.diff(sse_diffs)))] if len(sse_diffs) > 1 else K_range[0]
print(f"Suggested Elbow K = {elbow_k}")



# Load data
with open("mixture_model_data.pkl", "rb") as f:
    X = pickle.load(f)


print(type(data))
print(data.keys() if hasattr(data, "keys") else None)
print(data[:5] if isinstance(data, (list, tuple)) else None)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.metrics import pairwise_distances_argmin_min


def initialize_gmm(X, K, random_state=42):
    """Initialize means, covariances, and mixing coefficients for GMM."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    means = np.empty((K, d))
    means[0] = X[rng.integers(n)]
    dist_sq = np.sum((X - means[0]) ** 2, axis=1)
    for k in range(1, K):
        probs = dist_sq / dist_sq.sum()
        idx = rng.choice(n, p=probs)
        means[k] = X[idx]
        new_dist_sq = np.sum((X - means[k]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    covariances = np.array([np.cov(X.T) + np.eye(d) * 1e-6 for _ in range(K)])
    weights = np.ones(K) / K
    return means, covariances, weights


def gaussian_pdf(X, mean, cov):
    """Compute Gaussian density for each point."""
    n, d = X.shape
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm_const = 1 / np.sqrt((2 * np.pi) ** d * det)
    diff = X - mean
    exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
    return norm_const * exp_term


def expectation_step(X, means, covariances, weights):
    """E-step: compute responsibilities."""
    n, K = X.shape[0], len(weights)
    resp = np.zeros((n, K))
    for k in range(K):
        resp[:, k] = weights[k] * gaussian_pdf(X, means[k], covariances[k])
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp /= resp_sum
    return resp


def maximization_step(X, resp):
    """M-step: update means, covariances, and weights."""
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0)
    weights = Nk / n
    means = np.zeros((K, d))
    covariances = np.zeros((K, d, d))
    for k in range(K):
        means[k] = np.sum(resp[:, k][:, None] * X, axis=0) / Nk[k]
        diff = X - means[k]
        covariances[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
        covariances[k] += np.eye(d) * 1e-6
    return means, covariances, weights


def compute_log_likelihood(X, means, covariances, weights):
    """Compute total log-likelihood."""
    n, K = X.shape[0], len(weights)
    probs = np.zeros((n, K))
    for k in range(K):
        probs[:, k] = weights[k] * gaussian_pdf(X, means[k], covariances[k])
    total_prob = probs.sum(axis=1)
    return np.sum(np.log(total_prob + 1e-12))


def gmm_em(X, K, max_iter=200, tol=1e-6, random_state=42, verbose=False):
    """Run EM algorithm for Gaussian Mixture Models."""
    means, covariances, weights = initialize_gmm(X, K, random_state)
    log_likelihoods = []

    for i in range(max_iter):
        resp = expectation_step(X, means, covariances, weights)
        means, covariances, weights = maximization_step(X, resp)
        loglik = compute_log_likelihood(X, means, covariances, weights)
        log_likelihoods.append(loglik)

        if verbose:
            print(f"Iteration {i+1}: log-likelihood = {loglik:.6f}")

        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return means, covariances, weights, resp, log_likelihoods


def draw_conf2D(means, covariances, ax, n_std=2):
    """Draw ellipses representing each Gaussian component."""
    colors = ['r', 'g', 'b', 'orange', 'purple', 'k']
    for i in range(means.shape[0]):
        pearson = covariances[i, 0, 1] / np.sqrt(covariances[i, 0, 0] * covariances[i, 1, 1] + 1e-12)
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          edgecolor=colors[i % len(colors)],
                          lw=1.5, fill=False)
        scale_x = np.sqrt(covariances[i, 0, 0]) * n_std
        scale_y = np.sqrt(covariances[i, 1, 1]) * n_std
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(means[i, 0], means[i, 1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)


# -------------------------------------------------------
# Run GMM for a chosen K
# -------------------------------------------------------
K = 3
means, covariances, weights, resp, log_likelihoods = gmm_em(X, K, verbose=True)

# Plot log-likelihood progression
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.title(f"GMM Convergence (K={K})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize final clusters
labels = np.argmax(resp, axis=1)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
draw_conf2D(means, covariances, plt.gca(), n_std=2)
plt.title(f"GMM Clustering (K={K})")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Try multiple K values and compare BIC
# -------------------------------------------------------
def compute_bic(X, loglik, K):
    """Compute Bayesian Information Criterion."""
    n, d = X.shape
    p = K * d + K * (d * (d + 1) // 2) + (K - 1)
    return -2 * loglik + p * np.log(n)

K_range = range(1, 10)
bics = []
for k in K_range:
    _, _, _, _, loglik = gmm_em(X, k, random_state=42)
    bic_val = compute_bic(X, loglik[-1], k)
    bics.append(bic_val)

plt.figure(figsize=(6, 4))
plt.plot(K_range, bics, marker="o")
plt.xlabel("Number of components (K)")
plt.ylabel("BIC")
plt.title("BIC Method for Optimal K")
plt.grid(True)
plt.tight_layout()
plt.show()
