import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def gower_hierarchical_clustering(X, y, categorical_cols, numerical_cols, plot_dendrogram=False, initial_cluster_len=None, clusters=None):

    if initial_cluster_len is None:
        initial_cluster_len = len(X)

    if clusters is None:
        clusters = {}

    X_gower = X.copy()
    for col in numerical_cols:
        X_gower[col] = X_gower[col].astype(float)

    D = gower.gower_matrix(X_gower)
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    if plot_dendrogram:
        plt.figure(figsize=(12, 5))
        dendrogram(Z, truncate_mode="level", p=5)
        plt.show()

    # Use silhouette score to find the best number of clusters
    print(f"Initial cluster size: {len(X_gower)}")
    best_k, _ = silhouette_score_criterion(D)

    # Initialize clusters

    if best_k != -1 and best_k > 1:

        cluster_labels = fcluster(Z, t=best_k, criterion='maxclust')

        print(f"Cluster divided into: {best_k}", "clusters")
        # print(f"Cluster labels size: {len(cluster_labels)}")
        
        check_size = True
        total_cluster_size = len(X)

        for i in range(1, best_k + 1):
            X_cluster = X[cluster_labels == i]
            y_cluster = y[cluster_labels == i]
            print(f"Cluster {i} size: {len(X_cluster)}")


            # min_cluster_size = len(X_cluster)/total_cluster_size
            min_cluster_size = 1300


            if len(X_cluster) < min_cluster_size :
                check_size = False
                print("small cluster was found in:", {len(X_cluster)})


        print("----------------------------")

        if check_size:
            for i in range(1, best_k + 1):
                
                X_cluster = X[cluster_labels == i]
                y_cluster = y[cluster_labels == i]

                # If the cluster is large enough, recursively split it

                print("i am going to divide cluster:", i, "with size:", len(X_cluster))
                sub_clusters, _ = gower_hierarchical_clustering(X_cluster, y_cluster, categorical_cols, numerical_cols, plot_dendrogram=False, initial_cluster_len=initial_cluster_len, clusters=clusters)

        else:

            
            len1 =len(X[cluster_labels == 1])
            len2 =len(X[cluster_labels == 2])

            minimum = min(len1,len2)

            print("len1:", len1, "len2:", len2)
            #check which is the big cluster

            if len1/initial_cluster_len > 0.5 or len2/initial_cluster_len > 0.5:
                print("one dataframes assigned to clusters")

                if len1/initial_cluster_len > 0.5:
                    big_cluster_x =X[cluster_labels == 1]
                    big_cluster_y =y[cluster_labels == 1]
                    
                    print("clusters before:", len(clusters))
                    clusters[len(clusters) + 1] = (X[cluster_labels == 2], y[cluster_labels == 2])
                    print("clusters after:", len(clusters))

                elif len2/initial_cluster_len > 0.5:
                    big_cluster_x =X[cluster_labels == 2]
                    big_cluster_y =y[cluster_labels == 2]
                    
                    print("clusters before:", len(clusters))
                    clusters[len(clusters) + 1] = (X[cluster_labels == 1], y[cluster_labels == 1])
                    print("clusters after:", len(clusters))

                if big_cluster_x is not None:
                    sub_clusters, _ = gower_hierarchical_clustering(big_cluster_x, big_cluster_y, categorical_cols, numerical_cols, plot_dendrogram=False, initial_cluster_len=initial_cluster_len, clusters=clusters)
            else:   

                print("both dataframes assigned to clusters")
                print("clusters before:", len(clusters))


                total = len1 + len2
                ratio = minimum/total

                if ratio < 0.1:
                    mask = (cluster_labels == 1) | (cluster_labels == 2)
                    print("Note: Merging clusters because too small")
                    X_merged = X[mask]
                    y_merged = y[mask]
                    print("merged cluster size:", len(X_merged))
                    clusters[len(clusters) + 1] = (X_merged, y_merged)
                else: 
                    for i in range(1, best_k + 1):
                        X_sub = X[cluster_labels == i]
                        y_sub = y[cluster_labels == i]
                        clusters[len(clusters) + 1] = (X_sub, y_sub)
                print("clusters after:", len(clusters))

    else:
        clusters[1] = (X, y)

    return clusters, D



def silhouette_score_criterion(gower_matrix,show_results=False):
    '''
    silhouette_score_criterion uses silhouette score to determine the optimal number of clusters (k) for hierarchical clustering.
    
    param:
        gower_matrix: Gower distance matrix
    return: 
        print k as number of clusters and silhouette score for each k
    '''
   
    best_score= 0

    for k in range(2, 8):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )
        labels = model.fit_predict(gower_matrix)
        score = silhouette_score(gower_matrix, labels, metric="precomputed")
        if score > best_score:
            best_k = k
            best_score = score

        if show_results:
            print(f"silhouette analysis: k={k}, score={score}")

    if best_k >= 0.3:
        return best_k, best_score
    return -1, -1


def pca_mixed_data_visualization(clusters, categorical_cols, numerical_cols,visualization=False):
    

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    n_clusters = len(clusters)

    if visualization:
        n_cols = 2
        n_rows = int(np.ceil(n_clusters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

    for i, (X_cluster, y_cluster) in clusters.items():
        X_cluster, y_cluster = clusters[i]

        df = X_cluster.copy() 

        X_proc = preprocess.fit_transform(df)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_proc)


        if visualization:
            ax = axes[i - 1]

            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA mixed data visualization - Cluster {i}")

            ax.text(
                0.02, 0.98,
                f"n = {len(df)}",
                transform=ax.transAxes,
                verticalalignment="top"
            )

    if visualization:
        for j in range(i, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()