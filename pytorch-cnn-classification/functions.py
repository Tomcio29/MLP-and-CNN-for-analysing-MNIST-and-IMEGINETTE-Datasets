import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.neighbors import NearestNeighbors


def evaluate_kmeans(X, y_true, cluster_range):
    silhouette_scores = []
    ari_scores = []
    homogeneity_scores = []
    completeness_scores = []
    combined_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        ari_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))
        combined_score = (0.4 * ari_scores[-1] +
                          0.3 * completeness_scores[-1] +
                          0.3 * homogeneity_scores[-1])
        combined_scores.append(combined_score)

    return {
        "silhouette_scores": silhouette_scores,
        "ari_scores": ari_scores,
        "homogeneity_scores": homogeneity_scores,
        "completeness_scores": completeness_scores,
        "combined_scores": combined_scores,
        "best_k_silhouette": cluster_range[np.argmax(silhouette_scores)],
        "best_k_combined": cluster_range[np.argmax(combined_scores)],
        "worst_k_silhouette": cluster_range[np.argmin(silhouette_scores)],
        "worst_k_combined": cluster_range[np.argmin(combined_scores)],
    }

def evaluate_dbscan(X, y_true, eps_range):
    silhouette_scores = []
    ari_scores = []
    homogeneity_scores = []
    completeness_scores = []
    n_clusters_list = []
    valid_eps = []
    combined_scores = []

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        labels = dbscan.fit_predict(X)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        n_clusters = len(unique_labels)

        if n_clusters > 1 and n_clusters < len(X):
            # Sprawdź, czy każdy klaster ma co najmniej 2 punkty
            counts = [np.sum(labels == label) for label in unique_labels]
            if min(counts) < 2:
                continue  # pomiń ten eps, bo są klastry jednoelementowe

            sil = silhouette_score(X, labels)
            ari = adjusted_rand_score(y_true, labels)
            hom = homogeneity_score(y_true, labels)
            com = completeness_score(y_true, labels)

            silhouette_scores.append(sil)
            ari_scores.append(ari)
            homogeneity_scores.append(hom)
            completeness_scores.append(com)
            n_clusters_list.append(n_clusters)
            valid_eps.append(eps)
            combined_scores.append(0.4 * ari + 0.3 * hom + 0.3 * com)

    return {
        "silhouette_scores": silhouette_scores,
        "ari_scores": ari_scores,
        "homogeneity_scores": homogeneity_scores,
        "completeness_scores": completeness_scores,
        "n_clusters": n_clusters_list,
        "valid_eps": valid_eps,
        "combined_scores": combined_scores,
        "best_eps_silhouette": valid_eps[np.argmax(silhouette_scores)] if valid_eps else None,
        "best_eps_combined": valid_eps[np.argmax(combined_scores)] if valid_eps else None,
        "best_combined_score": max(combined_scores) if combined_scores else None,
        "worst_eps_silhouette": valid_eps[np.argmin(silhouette_scores)] if valid_eps else None,
        "worst_eps_combined": valid_eps[np.argmin(combined_scores)] if valid_eps else None,
    }

def plot_voronoi_with_labels(X, labels, true_labels, title=""):
    plt.figure(figsize=(6, 6))
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([[xx[i, j], yy[i, j]]])
            distances, indices = nn.kneighbors(point)
            Z[i, j] = labels[indices[0][0]]
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='coolwarm', s=50, edgecolors='k')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.pause(0.5)