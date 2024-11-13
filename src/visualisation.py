import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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