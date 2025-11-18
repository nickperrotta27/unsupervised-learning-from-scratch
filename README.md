Unsupervised Learning From Scratch

This repository implements classic unsupervised learning algorithms from the ground up, without using high-level ML libraries.
It includes:

K-Means & K-Means++ clustering 

Density-based / mixture-model learning 
(Gaussian Mixture Models + EM Algorithm)

The goal is to demonstrate clean, vectorized, mathematically correct implementations of widely used unsupervised learning techniques.

Project Structure
├── kmeans_data.pkl              
├── mixture_model_data.pkl      
├── model.py              # All code + experiments
└── README.md

Part 1 — K-Means Clustering (from scratch)
Implemented:

K-Means++ initialization

Full iterative clustering:

assign clusters via Euclidean distance

update centroids

handle empty clusters

Convergence criteria based on centroid shift

Within-cluster variance tracking

Silhouette score evaluation

Convergence visualization

Example:
centroids, labels, var_history = kmeans(X, K=3, random_state=42)

Visualization:
plt.plot(var_history)
plt.title("K-Means Convergence")
plt.xlabel("Iteration")
plt.ylabel("Within-Cluster Variance")

Part 2 — Gaussian Mixture Models (GMM) + EM Algorithm

Implemented:

Mixture of Gaussians fit using the Expectation–Maximization algorithm

Soft cluster assignments (responsibilities)

Parameter updates:

mean vectors

covariance matrices

mixture weights

Log-likelihood tracking

Convergence detection

Example:
gmm = GaussianMixtureScratch(K=3)
gmm.fit(X, max_iter=200)
labels = gmm.predict(X)

Diagnostics:

Log-likelihood vs iteration

Cluster visualization

Comparison against K-Means

Datasets
kmeans_data.pkl

Contains a 2D or high-dimensional dataset used for clustering with K-Means.

mixture_model_data.pkl

Used for fitting Gaussian mixture models or another density-based unsupervised method.

Both files contain NumPy arrays stored via pickle.

Requirements
numpy
matplotlib
scikit-learn    # only for silhouette score and sanity-checking
