import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from functions import evaluate_kmeans, evaluate_dbscan


def plot_metrics(x_values, results, x_label, dataset_name, method_name):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_values, results["silhouette_scores"], label="Silhouette score", marker='o')
    ax1.plot(x_values, results["ari_scores"], label="Adjusted Rand Index", marker='o')
    ax1.plot(x_values, results["homogeneity_scores"], label="Homogeneity score", marker='o')
    ax1.plot(x_values, results["completeness_scores"], label="Completeness score", marker='o')
    ax1.set_title(f"{method_name} metrics on {dataset_name}")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Score")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Jeśli DBSCAN, dodaj liczbę klastrów na drugiej osi Y
    if method_name == "DBSCAN" and "n_clusters" in results:
        ax2 = ax1.twinx()
        ax2.plot(x_values, results["n_clusters"], label="Liczba klastrów", marker='s', color='purple', linestyle='--')
        ax2.set_ylabel("Liczba klastrów", color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



def plot_pca_2d(X, labels_pred, labels_true, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pred, cmap='viridis', marker='o', alpha=0.6,
                          label='Predicted clusters')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_true, cmap='plasma', marker='x', label='True labels')
    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Clusters")
    plt.grid(True)
    plt.show()


def analyze_dataset(name, X, y_true):
    print(f"\n--- Analiza zbioru: {name} ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, 10)
    eps_range = np.arange(0.05, 2.0, 0.1)

    # KMeans
    kmeans_results = evaluate_kmeans(X_scaled, y_true, k_range)
    print(f"KMeans - najlepszy n_clusters wg silhouette: {kmeans_results['best_k_silhouette']}")
    print(f"KMeans - najlepszy n_clusters wg combined score: {kmeans_results['best_k_combined']}")
    plot_metrics(list(k_range), kmeans_results, "n_clusters", name, "KMeans")

    # DBSCAN
    dbscan_results = evaluate_dbscan(X_scaled, y_true, eps_range)
    if dbscan_results["valid_eps"]:
        print(f"DBSCAN - najlepszy eps wg silhouette: {dbscan_results['best_eps_silhouette']:.3f}")
        print(f"DBSCAN - najlepszy eps wg combined score: {dbscan_results['best_eps_combined']:.3f}")
        plot_metrics(dbscan_results["valid_eps"], dbscan_results, "eps", name, "DBSCAN")
    else:
        print("DBSCAN - brak sensownych klastrów dla testowanych eps")

    # Wizualizacja PCA
    best_k = kmeans_results['best_k_combined']
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    plot_pca_2d(X_scaled, labels_kmeans, y_true, f"{name} - KMeans clustering (k={best_k})")

    if dbscan_results["valid_eps"]:
        best_eps = dbscan_results['best_eps_combined']
        dbscan = DBSCAN(eps=best_eps, min_samples=1)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        plot_pca_2d(X_scaled, labels_dbscan, y_true, f"{name} - DBSCAN clustering (eps={best_eps:.3f})")
        best_eps_sil = dbscan_results['best_eps_silhouette']
        dbscan_sil = DBSCAN(eps=best_eps_sil, min_samples=1)
        labels_dbscan_sil = dbscan_sil.fit_predict(X_scaled)
        plot_pca_2d(X_scaled, labels_dbscan_sil, y_true, f"{name} - DBSCAN clustering (eps={best_eps_sil:.3f}, silhouette)")

    # Tabela wyników podsumowująca najlepsze modele
    summary = pd.DataFrame({
        "Method": ["KMeans", "DBSCAN"],
        "Best Param": [kmeans_results['best_k_combined'], dbscan_results['best_eps_combined']],
        "Silhouette Score": [
            kmeans_results["silhouette_scores"][k_range.index(kmeans_results['best_k_combined'])],
            dbscan_results["silhouette_scores"][
                dbscan_results["valid_eps"].index(dbscan_results['best_eps_combined'])] if dbscan_results[
                "valid_eps"] else np.nan,
        ],
        "ARI": [
            kmeans_results["ari_scores"][k_range.index(kmeans_results['best_k_combined'])],
            dbscan_results["ari_scores"][dbscan_results["valid_eps"].index(dbscan_results['best_eps_combined'])] if
            dbscan_results["valid_eps"] else np.nan,
        ],
        "Homogeneity": [
            kmeans_results["homogeneity_scores"][k_range.index(kmeans_results['best_k_combined'])],
            dbscan_results["homogeneity_scores"][
                dbscan_results["valid_eps"].index(dbscan_results['best_eps_combined'])] if dbscan_results[
                "valid_eps"] else np.nan,
        ],
        "Completeness": [
            kmeans_results["completeness_scores"][k_range.index(kmeans_results['best_k_combined'])],
            dbscan_results["completeness_scores"][
                dbscan_results["valid_eps"].index(dbscan_results['best_eps_combined'])] if dbscan_results[
                "valid_eps"] else np.nan,
        ],
        "Combined Score": [
            kmeans_results["combined_scores"][k_range.index(kmeans_results['best_k_combined'])],
            dbscan_results["combined_scores"][dbscan_results["valid_eps"].index(dbscan_results['best_eps_combined'])] if
            dbscan_results["valid_eps"] else np.nan,
        ],
    })
    print("\nPodsumowanie najlepszych modeli:")
    print(summary.round(3).to_markdown(index=False))
    return summary

if __name__ == "__main__":
    # Wczytanie zbiorów
    iris = load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()

    # Uruchomienie analizy
    analyze_dataset("Iris", iris.data, iris.target)
    analyze_dataset("Wine", wine.data, wine.target)
    analyze_dataset("Breast Cancer Wisconsin", breast_cancer.data, breast_cancer.target)
