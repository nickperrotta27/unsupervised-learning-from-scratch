# Clustering Analysis with K-Means and Gaussian Mixture Models

## Overview

This project implements and visualizes clustering using two popular methods:

- **K-Means Clustering** – Assigns each point to exactly one cluster based on Euclidean distance.
- **Gaussian Mixture Models (GMM)** – Fits a mixture of Gaussian distributions to the data using the Expectation-Maximization (EM) algorithm, allowing soft cluster assignments.

The project includes evaluation and selection of the optimal number of clusters using Elbow, Silhouette, and BIC methods.

## Features

### K-Means

- K-Means++ style centroid initialization
- Cluster assignment and centroid update
- Convergence monitoring via average within-cluster variance
- Elbow and Silhouette methods to suggest optimal K
- Visualization of clusters with centroids

### Gaussian Mixture Models

- EM algorithm for fitting multivariate Gaussians
- Soft assignments (responsibilities) for each data point
- Log-likelihood monitoring for convergence
- Bayesian Information Criterion (BIC) for model selection
- Visualization of Gaussian components with 2D ellipses

## Installation

Clone the repository and install required packages:

```bash
pip install numpy matplotlib scikit-learn
```

The project is designed to run in Google Colab for easy interaction with `.pkl` data files.

## Usage

### Upload Data

Provide your data as a `.pkl` file containing a NumPy array `X` of shape `(n_samples, n_features)`.

### K-Means

```python
from kmeans_module import kmeans, assign_clusters

K = 3
centroids, labels, var_history = kmeans(X, K, verbose=True)
```

Visualize convergence:

```python
plt.plot(var_history)
plt.show()
```

Visualize clusters:

```python
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='X')
plt.show()
```

### Gaussian Mixture Models

```python
from gmm_module import gmm_em

K = 3
means, covariances, weights, resp, log_likelihoods = gmm_em(X, K, verbose=True)
labels = np.argmax(resp, axis=1)
```

Visualize clusters with ellipses representing Gaussian components:

```python
draw_conf2D(means, covariances, plt.gca())
plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()
```

## Optimal K Selection

- **K-Means**: Elbow (SSE) and Silhouette methods
- **GMM**: Bayesian Information Criterion (BIC)

## Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn` (only for silhouette score and sanity-checking)

## References

- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- Scikit-learn documentation: https://scikit-learn.org
