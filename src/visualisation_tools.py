import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

class Visualisation():

    @staticmethod
    def create_boxplot(data, x_column, y_column=None, title='Boxplot', figsize=(8, 6)):
        """
        Create boxplot visualization.

        Args:
            data (DataFrame): The dataset containing the data.
            x_column (str): The column to be plotted on the x-axis.
            y_column (str, optional): The column to be plotted on the y-axis. Default is None.
            title (str): The title of the boxplot. Default is 'Boxplot'.
            figsize (tuple): The size of the figure. Default is (8, 6).
        """
        plt.figure(figsize=figsize)
        if y_column:
            sns.boxplot(x=data[x_column], y=data[y_column])
        else:
            sns.boxplot(x=data[x_column])
        plt.title(title)
        plt.xlabel(x_column)
        if y_column:
            plt.ylabel(y_column)
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

    @staticmethod
    def create_scatter_plot(df, x_feature, y_feature, title='Scatter Plot', color='b', alpha=0.7, figsize=(10, 6)):
        """
        Plots a scatter plot for two numerical features.

        Args:
            df (DataFrame): The dataframe containing the data.
            x_feature (str): The column name for the x-axis.
            y_feature (str): The column name for the y-axis.
            title (str): The title of the scatter plot. Default is 'Scatter Plot'.
            color (str): Color of the points in the scatter plot.
            alpha (float): Transparency of the points (default is 0.7).
            figsize (tuple): The size of the figure (default is (10, 6)).
        """
        plt.figure(figsize=figsize)
        plt.scatter(df[x_feature], df[y_feature], color=color, alpha=alpha)
        plt.title(title)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.grid(True)
        plt.show()

    @staticmethod
    def create_heat_map(df, x_feature, y_feature, title='Heat Map', figsize=(10, 6)):
        """
        Creates a heatmap for the relationship between two features.

        Args:
            df (DataFrame): The dataframe containing the data.
            x_feature (str): The column name for the x-axis.
            y_feature (str): The column name for the y-axis.
            title (str): The title of the heat map. Default is 'Heat Map'.
            figsize (tuple): The size of the figure (default is (10, 6)).
        """
        plt.figure(figsize=figsize)
        heatmap_data = pd.crosstab(df[y_feature], df[x_feature])
        sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='g')
        plt.title(title)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()

    @staticmethod
    def create_violin_plot(data, x_column, y_column=None, title='Violin Plot', figsize=(8, 6)):
        """
        Create violin plot visualization.

        Args:
            data (DataFrame): The dataset containing the data.
            x_column (str): The column to be plotted on the x-axis.
            y_column (str, optional): The column to be plotted on the y-axis. Default is None.
            title (str): The title of the violin plot. Default is 'Violin Plot'.
            figsize (tuple): The size of the figure. Default is (8, 6).
        """
        plt.figure(figsize=figsize)
        if y_column:
            sns.violinplot(x=data[x_column], y=data[y_column])
        else:
            sns.violinplot(x=data[x_column])
        plt.title(title)
        plt.xlabel(x_column)
        if y_column:
            plt.ylabel(y_column)
        plt.show()

    @staticmethod
    def p_value_calculator(df, group1, group2):
        """
        Calculate p value test.
        
        Args:
            df (DataFrame): The dataset containing the data.
            group1 (str): The column of data group for p value test.
            group2 (str): The column of data group fro p value test.
        """

        group_1 = df[group1]
        group_2 = df[group2]

        # Calculate Pearson correlation and p-value
        correlation, p_value = stats.pearsonr(group_1, group_2)

        print(f"Correlation: {correlation}")
        print(f"p-value: {p_value}")

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: Significant correlation.")
        else:
            print("Fail to reject the null hypothesis: No significant correlation.")

    @staticmethod
    def chi_square_contribution(df, feature1, feature2):
        """
        Calculate and plot Chi-square contributions for the relationship between two categorical features.

        Args:
            df (DataFrame): The dataframe containing the data.
            feature1 (str): The first categorical feature.
            feature2 (str): The second categorical feature.
        """
        # Create a cross-tabulation
        crosstab = pd.crosstab(df[feature1], df[feature2])

        # Perform Chi-square test
        chi2_stat, p, dof, expected = stats.chi2_contingency(crosstab)

        # Calculate Chi-square contributions
        chi2_contributions = (crosstab - expected) ** 2 / expected

        # Plot the contributions
        plt.figure(figsize=(10, 6))
        sns.heatmap(chi2_contributions, annot=True, cmap='Reds', fmt='.2f')
        plt.title('Chi-Square Contribution Heatmap')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()

        print(f"Chi-square statistic: {chi2_stat}")
        print(f"p-value: {p}")
        if p < 0.05:
            print("There is a significant relationship between the features.")
        else:
            print("There is no significant relationship between the features.")

    @staticmethod
    def anova_test(df, numerical_variable, categorical_variable):
        """
        Perform an ANOVA test to determine if there is a significant difference in the means of a numerical variable across different categories.

        Args:
            df (DataFrame): The dataframe containing the data.
            numerical_variable (str): The column name of the numerical variable to be tested.
            categorical_variable (str): The column name of the categorical variable to group by.

        Prints:
            F-statistic and p-value from the ANOVA test.
            A message indicating whether there is a significant difference in means across the categories.
        """

        # Separate the numerical data by category
        groups = [df[df[categorical_variable] == category][numerical_variable] for category in df[categorical_variable].unique()]

        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(*groups)

        print(f"F-statistic: {f_stat}, p-value: {p_value}")

        if p_value < 0.05:
            print("There is a significant difference in means across the categories.")
        else:
            print("There is no significant difference in means across the categories.")