import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Load the digits dataset
data, labels = load_digits(return_X_y=True)

# Choose the 3rd, 4th, and 5th digits
selected_digits = [2, 3, 4]
indices = np.isin(labels, selected_digits)
chosen_samples, chosen_labels = data[indices], labels[indices]

# Dimensionality reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(chosen_samples)

# K-means algorithm
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(reduced_data)
    inertias.append(kmeans.inertia_)

# Elbow method graph
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow method for K-means')
plt.show()


# After looking at the graph and the elbow method, we can clearly see that
# the optimal k is equal to 3
optimal_k = 3
kmeans_optimal = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
kmeans_labels = kmeans_optimal.fit_predict(reduced_data)


# Define a custom cost function for K-means
def k_means_cost(data_points, centroids, labels):
    cost = 0.0
    for i, point in enumerate(data_points):
        centroid = centroids[labels[i]]
        cost += np.sum((point - centroid) ** 2)
    return cost


# Calculate the cost using the custom cost function
centroids = kmeans_optimal.cluster_centers_
cost = k_means_cost(reduced_data, centroids, kmeans_labels)
print(f"Custom cost (inertia) for K={optimal_k}: {cost}")
# we tested to see the cost of the k means algorithm with the optimal k that
# we chose to be 3 and got the result to be 39250,
# we also noticed that for k = 2 it was more than double

K = 3
# N is the number of samples, D is the number of features
N, D = reduced_data.shape
pi = np.ones(K) / K
mu = reduced_data[np.random.choice(N, K, replace=False)]  # random means

sigma = np.zeros((K, D, D))

for k in range(K):
    sigma[k] = np.diag(np.var(reduced_data, axis=0))


# Helper functions
def gaussian_pdf(x, mu, sigma):
    norm = 1 / (np.linalg.det(sigma) ** 0.5 * (2 * np.pi) ** (D / 2))
    exp = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1))
    return norm * exp


def log_likelihood(X, pi, mu, sigma):
    N = X.shape[0]
    ll = 0
    for i in range(N):
        ll += np.log(sum(pi[k] * gaussian_pdf(X[i:i+1], mu[k], sigma[k]) for k in range(K)))
    return ll


# GMM algorithm
max_iter = 100
tolerance = 1e-6
prev_ll = -np.inf
ll_history = []

for iteration in range(max_iter):

    responsibilities = np.zeros((N, K))
    for k in range(K):
        responsibilities[:, k] = pi[k] * gaussian_pdf(reduced_data, mu[k], sigma[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    Nk = responsibilities.sum(axis=0)
    pi = Nk / N
    for k in range(K):
        mu[k] = np.sum(responsibilities[:, k:k+1] * reduced_data, axis=0) / Nk[k]
        diff = reduced_data - mu[k]
        sigma[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]

    # Compute log-likelihood
    ll = log_likelihood(reduced_data, pi, mu, sigma)
    ll_history.append(ll)

    # Check for convergence
    if np.abs(ll - prev_ll) < tolerance:
        break
    prev_ll = ll

    print(f"Iteration {iteration + 1}, Log-Likelihood: {ll}")

# Plot log-likelihood history
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(ll_history) + 1), ll_history)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Iteration')
plt.show()
plt.savefig('log_likelihood.png')

# Final clustering
final_responsibilities = np.zeros((N, K))
for k in range(K):
    final_responsibilities[:, k] = pi[k] * gaussian_pdf(reduced_data, mu[k], sigma[k])
final_responsibilities /= final_responsibilities.sum(axis=1, keepdims=True)
cluster_assignments = np.argmax(final_responsibilities, axis=1)

# Visualize the clustering results
plt.figure(figsize=(10, 5))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap='viridis')
plt.title('GMM Clustering')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
plt.savefig('gmm_clustering.png')


def compute_accuracy(true_labels, predicted_clusters):
    # Create a confusion matrix
    conf_mat = np.zeros((K, K))
    for i in range(len(true_labels)):
        conf_mat[true_labels[i], predicted_clusters[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-conf_mat)

    cluster_to_label = {col: row for row, col in zip(row_ind, col_ind)}

    mapped_predictions = np.array([cluster_to_label[cluster] for cluster in predicted_clusters])

    # Compute accuracy
    accuracy = accuracy_score(true_labels, mapped_predictions)

    return accuracy


# We need to map our chosen_labels to 0, 1, 2
label_mapping = {2: 0, 3: 1, 4: 2}
true_labels = np.array([label_mapping[label] for label in chosen_labels])

# Compute accuracy
gmm_accuracy = compute_accuracy(true_labels, cluster_assignments)
print(f"GMM Clustering Accuracy: {gmm_accuracy:.4f}")
# we got an accuracy of 0.9279
