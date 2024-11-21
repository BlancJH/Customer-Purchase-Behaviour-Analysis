from sklearn.cluster import KMeans

class FeatureEngeeniering():

    @staticmethod
    def k_means_cluster_bin(df, feature, cluster_number, random_state=42):


        # Reshape data for KMeans
        feature_values = df[feature].values.reshape(-1, 1)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=cluster_number, random_state=random_state)
        df[f'{feature}_cluster'] = kmeans.fit_predict(feature_values)

        # Generate cluster labels dynamically based on the cluster_number
        cluster_labels = {i: f"Cluster {i + 1}" for i in range(cluster_number)}

        # Assign cluster labels based on K-means
        df[f'{feature}_binned'] = df[f'{feature}_cluster'].map(cluster_labels)
        
        return df