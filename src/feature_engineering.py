from sklearn.cluster import KMeans
import numpy as np

class FeatureEngineering():

    @staticmethod
    def k_means_cluster_bin(df, feature, cluster_number, random_state=42):
        """
        Apply K-means clustering to bin a numerical feature into clusters and create cluster labels.

        Args:
            df (DataFrame): The DataFrame containing the data.
            feature (str): The column name of the numerical feature to be clustered.
            cluster_number (int): The number of clusters to create.
            random_state (int, optional): The random state for reproducibility. Default is 42.

        Returns:
            DataFrame: The DataFrame with new columns added for cluster labels and cluster names.
                - '{feature}_cluster': The cluster label for each row.
                - '{feature}_binned': The descriptive cluster label for each row.
        """


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
        """
        Calculate the count and percentage of a categorical feature within each group defined by another feature.

        Args:
            df (DataFrame): The DataFrame containing the data.
            feature (str): The column name of the categorical feature to be analyzed.
            groupby_feature (str): The column name of the feature to group by (e.g., clusters).

        Prints:
            DataFrame: A DataFrame containing the count and percentage of each category within each group.
                - '{groupby_feature}': The group (e.g., cluster).
                - '{feature}': The categorical feature.
                - 'count': The count of each category within each group.
                - 'total_count': The total count for each group.
                - 'percentage': The percentage of each category within each group.
        """
        
        # Calculate the count of each category within each cluster
        feature_counts = df.groupby([f'{groupby_feature}', f'{feature}']).size().reset_index(name='count')

        # Calculate the total count for each cluster
        total_counts = df.groupby(f'{groupby_feature}')[f'{feature}'].count().reset_index(name='total_count')

        # Merge to get the total count per cluster for percentage calculation
        feature_counts = feature_counts.merge(total_counts, on=f'{groupby_feature}')

        # Calculate percentage
        feature_counts['percentage'] = (feature_counts['count'] / feature_counts['total_count']) * 100

        print(feature_counts)


    @staticmethod
    def identify_outliers(data, x_column):
        """
        Identify and return outliers in the specified column.

        Args:
            data (DataFrame): The dataset containing the data.
            x_column (str): The column to check for outliers.

        Returns:
            Tuple: A tuple containing:
                - Series: A pandas Series containing the outlier values.
                - float: The lower bound used for outlier detection.
                - float: The upper bound used for outlier detection.
        """
        q1 = np.percentile(data[x_column], 25)
        q3 = np.percentile(data[x_column], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[x_column] < lower_bound) | (data[x_column] > upper_bound)][x_column]
        return outliers, lower_bound, upper_bound