import numpy as np
import scipy.sparse as sp

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans as SKLearnKMeans


'''
# Load Dataset
'''

categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

'''
# Evaluating fitness
'''
def fit_and_evaluate(km, X, n_runs=5):

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

'''
# Vectorizes dataset using tfidf
'''

vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)

'''
# Custom K-Means Clustering Implementation
'''

class KMeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    def fit(self, X):

        # Initialize centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices].toarray()  # Convert selected centroids to dense format

        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()

            # Calculate distances using pairwise euclidean distances
            distances = euclidean_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)

            # Update centroids
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]

                # Check if cluster is empty
                if cluster_points.shape[0] > 0:
                    # Compute mean for sparse matrix
                    self.centroids[k] = np.mean(cluster_points.toarray(), axis=0)

            # Check for convergence (using a threshold for floating-point comparison)
            if np.allclose(old_centroids, self.centroids, atol=1e-7):
                break

    def set_params(self, random_state):
        np.random.seed(random_state)

# Call my own KMeans and evaluate for fitness    
kmeans = KMeans(n_clusters=true_k, max_iter=500)
fit_and_evaluate(kmeans, X_tfidf)

# Use SKLearn's built-in Kmeans to compare to my results
sklearn_kmeans = SKLearnKMeans(n_clusters=true_k, max_iter=300, random_state=42, n_init=1)
sklearn_kmeans.fit(X_tfidf)
fit_and_evaluate(sklearn_kmeans, X_tfidf, n_runs=5)
