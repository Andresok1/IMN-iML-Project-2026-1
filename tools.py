import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

def gower_hierarchical_clustering(X, y, categorical_cols, numerical_cols,  plot_dendrogram=False):

    X_gower = X.copy()

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_gower)

    for col in numerical_cols:
        X_gower[col] = X_gower[col].astype(float)

    D = gower.gower_matrix(X_gower)
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    if plot_dendrogram:
        plt.figure(figsize=(12, 5))
        dendrogram(Z, truncate_mode="level", p=5)
        plt.show()

    best_k, _= silhouette_score_criterion(D)

    cluster_labels = fcluster(Z, t=best_k, criterion='maxclust')

    # print(f"Number of clusters: {best_k}")
    # print(f"Cluster labels size: {len(cluster_labels)}")
    # print(f"len(X) labels size: {len(X_gower)}")

    X_cluster_0 = X[cluster_labels == 1] 
    y_cluster_0 = y[cluster_labels == 1] 

    X_cluster_1 = X[cluster_labels == 2]  
    y_cluster_1 = y[cluster_labels == 2]     

    clusters = {i: (X[cluster_labels == i], y[cluster_labels == i]) for i in range(1, best_k + 1)}


    return clusters, D


def silhouette_score_criterion(gower_matrix):
    '''
    silhouette_score_criterion uses silhouette score to determine the optimal number of clusters (k) for hierarchical clustering.
    
    param:
        gower_matrix: Gower distance matrix
    return: 
        print k as number of clusters and silhouette score for each k
    '''

    for k in range(2, 8):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )
        best_score= 0
        labels = model.fit_predict(gower_matrix)
        score = silhouette_score(gower_matrix, labels, metric="precomputed")
        if score > best_score:
            best_k = k
            best_score = score

        print(f"silhouette analysis: k={k}, score={score}")
        return best_k, best_score

    