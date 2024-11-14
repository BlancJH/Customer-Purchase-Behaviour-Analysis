import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualisation():

    @staticmethod
    def create_boxplot(data, x_column, title='Boxplot', figsize=(8, 6)):
        """
        Create boxplot visualisation.

        Args:
            data (DataFrame): The dataset containing the data.
            x_column (str): The column to be plotted.
            title (str): The title of the boxplot. Default is 'Boxplot'.
            figsize (tuple): The size of the figure. Default is (8, 6).
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x=data[x_column])
        plt.title(title)
        plt.xlabel(x_column)
        plt.show()

    @staticmethod
    def identify_outliers(data, x_column):
        """
        Identify and return outliers in the specified column.

        Args:
            data (DataFrame): The dataset containing the data.
            x_column (str): The column to check for outliers.

        Returns:
            Series: A pandas Series containing the outlier values.
        """
        q1 = np.percentile(data[x_column], 25)
        q3 = np.percentile(data[x_column], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[x_column] < lower_bound) | (data[x_column] > upper_bound)][x_column]
        return outliers

    @staticmethod
    def create_histogram(df, feature, bins=30, color='skyblue'):
        """
        Plots a histogram for a given feature.

        Args:
            df (DataFrame): The dataframe containing the data.
            feature (str): The column name for which the histogram will be plotted.
            bins (int): Number of bins for the histogram.
            color (str): Color of the bars in the histogram.
        """
        plt.figure(figsize=(10, 5))
        plt.hist(df[feature], bins=bins, color=color, edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def create_bar_chart(df, feature, color='skyblue'):
        """
        Plots a bar chart for a given categorical feature.

        Args:
            df (DataFrame): The dataframe containing the data.
            feature (str): The column name for which the bar chart will be plotted.
            color (str): Color of the bars in the bar chart.
        """
        plt.figure(figsize=(10, 5))
        sns.countplot(x=feature, data=df, color=color, edgecolor='black')
        plt.title(f'Count of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def create_pie_chart(df, feature):
        """
        Plots a pie chart for a given categorical feature.

        Args:
            df (DataFrame): The dataframe containing the data.
            feature (str): The column name for which the pie chart will be plotted.
        """
        plt.figure(figsize=(7, 7))
        df[feature].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
        plt.title(f'Proportion of {feature}')
        plt.ylabel('')
        plt.show()