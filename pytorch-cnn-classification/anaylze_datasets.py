import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
from functions import evaluate_kmeans, evaluate_dbscan
from functions import plot_voronoi_with_labels
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path, sep=";", header=None)
    X = df.iloc[:, :2].values  # cechy (X1, X2)
    y_true = df.iloc[:, 2].values  # etykiety
    return X, y_true
def plot_kmeans_metrics(cluster_range, kmeans_results, dataset_path):
    plt.figure(figsize=(6, 5))
    plt.plot(cluster_range, kmeans_results["silhouette_scores"], marker='o', label="Silhouette Score")
    plt.title(f"KMeans – silhouette score ({dataset_path})")
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(0.5)

    plt.figure(figsize=(6, 5))
    plt.plot(cluster_range, kmeans_results["ari_scores"], marker='o', label="ARI", color='blue')
    plt.plot(cluster_range, kmeans_results["homogeneity_scores"], marker='o', label="Homogeneity", color='green')
    plt.plot(cluster_range, kmeans_results["completeness_scores"], marker='o', label="Completeness", color='red')
    plt.title(f"KMeans – miary ({dataset_path})")
    plt.xlabel("n_clusters")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(0.5)


def plot_dbscan_metrics(dbscan_results, dataset_path):
    if dbscan_results["valid_eps"]:
        plt.figure(figsize=(8, 6))
        plt.plot(dbscan_results["valid_eps"], dbscan_results["silhouette_scores"], marker='o', label="Silhouette Score", color='purple')
        for eps_val, sil_score, n_clusters in zip(dbscan_results["valid_eps"], dbscan_results["silhouette_scores"], dbscan_results["n_clusters"]):
            plt.text(eps_val, sil_score + 0.01, f"{n_clusters}", ha='center', va='bottom', fontsize=11)
        plt.title(f"DBSCAN – Silhouette Score ({dataset_path})")
        plt.xlabel("eps")
        plt.ylabel("Silhouette score")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.pause(0.5)

        plt.figure(figsize=(6, 5))
        plt.plot(dbscan_results["valid_eps"], dbscan_results["ari_scores"], marker='o', label="ARI", color='blue')
        plt.plot(dbscan_results["valid_eps"], dbscan_results["homogeneity_scores"], marker='o', label="Homogeneity", color='green')
        plt.plot(dbscan_results["valid_eps"], dbscan_results["completeness_scores"], marker='o', label="Completeness", color='red')
        for eps_val, n_clusters in zip(dbscan_results["valid_eps"], dbscan_results["n_clusters"]):
            plt.text(eps_val, 0.1, f"{n_clusters}", ha='center', va='top', fontsize=11)
        plt.title(f"DBSCAN – miary ({dataset_path})")
        plt.xlabel("eps")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.pause(0.5)
    else:
        print("Brak dobrych eps dla DBSCAN (więcej niż 1 klaster)")

def analyze_cluster_data(dataset_path, cluster_range=range(2, 10), eps_range=np.arange(0.05, 2.0, 0.1)):
    X, y_true = load_dataset(dataset_path)
    kmeans_results = evaluate_kmeans(X, y_true, cluster_range)
    dbscan_results = evaluate_dbscan(X, y_true, eps_range)

    plot_kmeans_metrics(cluster_range, kmeans_results, dataset_path)
    plot_dbscan_metrics(dbscan_results, dataset_path)

    # --- Wyniki tekstowe ---
    print(f"\n=== KMeans vs y_true ({dataset_path}) ===")
    print(f"Best Silhouette Score (for KMeans): {max(kmeans_results['silhouette_scores'])}")
    print(f"Best k (n_clusters) based on Silhouette Score: {kmeans_results['best_k_silhouette']}")
    print(f"Best k (n_clusters) based on Combined Score: {kmeans_results['best_k_combined']}")
    best_k_index = np.argmax(kmeans_results["combined_scores"])
    print(f"--- Metrics for best k (n_clusters = {kmeans_results['best_k_combined']}) ---")
    print(f"ARI: {kmeans_results['ari_scores'][best_k_index]}")
    print(f"Homogeneity: {kmeans_results['homogeneity_scores'][best_k_index]}")
    print(f"Completeness: {kmeans_results['completeness_scores'][best_k_index]}")
    print(f"Combined Score: {kmeans_results['combined_scores'][best_k_index]}")

    if dbscan_results["valid_eps"]:
        best_eps_combined = dbscan_results["best_eps_combined"]
        dbscan_best_combined = DBSCAN(eps=best_eps_combined, min_samples=1).fit(X)
        dbscan_labels_combined = dbscan_best_combined.labels_
        print(f"\n=== DBSCAN vs y_true ({dataset_path}) ===")
        print(f"Best Silhouette Score (for DBSCAN): {max(dbscan_results['silhouette_scores'])}")
        print(f"Best eps (by Silhouette): {dbscan_results['best_eps_silhouette']}")
        print(f"ARI (best eps): {adjusted_rand_score(y_true, dbscan_labels_combined)}")
        print(f"Homogeneity (best eps): {homogeneity_score(y_true, dbscan_labels_combined)}")
        print(f"Completeness (best eps): {completeness_score(y_true, dbscan_labels_combined)}")
        print(f"Best eps (by Combined Score): {best_eps_combined}")
        print(f"Combined Score (best eps): {dbscan_results['best_combined_score']}")
    else:
        print(f"\n=== DBSCAN vs y_true ({dataset_path}) ===")
        print("Brak dobrych eps dla DBSCAN (więcej niż 1 klaster)")

    # --- Diagramy Woronoja ---
    kmeans_best_silhouette = KMeans(n_clusters=kmeans_results["best_k_silhouette"], random_state=42).fit(X)
    plot_voronoi_with_labels(X, kmeans_best_silhouette.labels_, y_true,
                             f"KMeans (n_clusters={kmeans_results['best_k_silhouette']}) – najlepszy Silhouette Score ({dataset_path})")

    kmeans_best_combined = KMeans(n_clusters=kmeans_results["best_k_combined"], random_state=42).fit(X)
    plot_voronoi_with_labels(X, kmeans_best_combined.labels_, y_true,
                             f"KMeans (n_clusters={kmeans_results['best_k_combined']}) – najlepszy Combined Score ({dataset_path})")
    kmeans_worst_silhouette = KMeans(n_clusters=kmeans_results["worst_k_silhouette"], random_state=42).fit(X)
    plot_voronoi_with_labels(X, kmeans_worst_silhouette.labels_, y_true,
                            f"KMeans (n_clusters={kmeans_results['worst_k_silhouette']}) – NAJGORSZY Silhouette Score ({dataset_path})")

    kmeans_worst_combined = KMeans(n_clusters=kmeans_results["worst_k_combined"], random_state=42).fit(X)
    plot_voronoi_with_labels(X, kmeans_worst_combined.labels_, y_true,
                            f"KMeans (n_clusters={kmeans_results['worst_k_combined']}) – NAJGORSZY Combined Score ({dataset_path})")

    if dbscan_results["valid_eps"]:
        dbscan_best_silhouette = DBSCAN(eps=dbscan_results["best_eps_silhouette"], min_samples=1).fit(X)
        plot_voronoi_with_labels(X, dbscan_best_silhouette.labels_, y_true,
                                 f"DBSCAN (eps={dbscan_results['best_eps_silhouette']:.2f}) – najlepszy Silhouette Score ({dataset_path})")
        dbscan_best_combined = DBSCAN(eps=dbscan_results["best_eps_combined"], min_samples=1).fit(X)
        plot_voronoi_with_labels(X, dbscan_best_combined.labels_, y_true,
                                 f"DBSCAN (eps={dbscan_results['best_eps_combined']:.2f}) – najlepszy Combined Score ({dataset_path})")
        dbscan_worst_silhouette = DBSCAN(eps=dbscan_results["worst_eps_silhouette"], min_samples=1).fit(X)
        plot_voronoi_with_labels(X, dbscan_worst_silhouette.labels_, y_true,
                                 f"DBSCAN (eps={dbscan_results['worst_eps_silhouette']:.2f}) – NAJGORSZY Silhouette Score ({dataset_path})")

        dbscan_worst_combined = DBSCAN(eps=dbscan_results["worst_eps_combined"], min_samples=1).fit(X)
        plot_voronoi_with_labels(X, dbscan_worst_combined.labels_, y_true,
                                 f"DBSCAN (eps={dbscan_results['worst_eps_combined']:.2f}) – NAJGORSZY Combined Score ({dataset_path})")

# ==========================
# Analiza dla wszystkich zbiorów
# ==========================
def main():
    for i in range(1, 4):
        analyze_cluster_data(f"datasets/2_{i}.csv")
    for i in range(1, 4):
        analyze_cluster_data(f"datasets/3_{i}.csv")
        n_clusters = 2

if __name__ == "__main__":
    main()
