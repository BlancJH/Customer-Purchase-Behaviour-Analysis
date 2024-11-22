from sklearn.cluster import KMeans

class FeatureEngineering():

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
    
    @staticmethod
    def feature_grouping_count(df, feature, groupby_feature):
        
        # Calculate the count of each category within each cluster
        feature_counts = df.groupby([f'{groupby_feature}', f'{feature}']).size().reset_index(name='count')

        # Calculate the total count for each cluster
        total_counts = df.groupby(f'{groupby_feature}')[f'{feature}'].count().reset_index(name='total_count')

        # Merge to get the total count per cluster for percentage calculation
        feature_counts = feature_counts.merge(total_counts, on=f'{groupby_feature}')

        # Calculate percentage
        feature_counts['percentage'] = (feature_counts['count'] / feature_counts['total_count']) * 100

        print(feature_counts)