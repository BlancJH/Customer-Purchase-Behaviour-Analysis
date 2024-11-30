# Customer Purchase Behaviour Analysis
![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

## Project Overview

This project aims to understand customer purchasing behaviour to improve marketing strategies and enhance business decision-making. Using the 'Customer Purchases Behaviour Dataset' (Goyal, 2021), the analysis focused on identifying factors influencing purchasing decisions and predicting customers likely to subscribe to new products. The findings can assist in creating targeted marketing campaigns and optimising product offerings.

## Dataset

The dataset used in this project is the 'Customer Purchases Behaviour Dataset' from Kaggle, which is AI-generated to mimic real-world commercial data. It contains 10,000 records and 12 attributes, including customer demographics (age, gender, income, education, etc.) and purchasing history (purchase amount, product category, promotion usage, etc.).

## Project Structure

notebooks/: Contains Jupyter notebooks used for Exploratory Data Analysis (EDA) and model experimentation.

src/: Source code files for data processing, feature engineering, and model building.

data/: Raw dataset files.

models/: Saved models used for predicting customer purchasing behaviour.

reports/: Contains figures and final analysis report.

## Methodology

Data Cleaning and Preprocessing: The dataset was cleaned to handle outliers, and relevant features were selected. Missing values were not detected in the AI-generated dataset.

Exploratory Data Analysis (EDA): EDA was conducted to understand the structure and distribution of data. Key patterns, such as the correlation between income level and purchase amount, were identified.

Model Development: Four models were tested to predict purchase amount based on customer income: Linear Regression, Polynomial Regression, Decision Tree, and Random Forest. These models were evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².

Insights: The income level was identified as the key predictor of purchase amount. Customer segmentation by education, region, and age provided insights for targeted marketing strategies.

## Key Findings

Income level is the strongest predictor of purchasing behaviour, with higher-income customers making larger purchases.

The Polynomial Regression model outperformed other models in predicting purchasing amounts.

The electronics and clothing categories dominate sales, suggesting a focus on these areas could maximise revenue.

Customers primarily belong to the mid-20s to mid-30s age group, providing a key target for future marketing efforts.

## Limitations

The dataset was AI-generated and evenly distributed, limiting the discovery of some meaningful real-world relationships. Future work should focus on using real-world data to uncover more actionable insights.

How to Use This Project

Install Dependencies: Use the requirements.txt file to install all the necessary libraries.

Run Analysis: Start by running the Jupyter notebooks for EDA and model experiments.

## Future Work

Create end-to-end pipeline to import data and make predictions.

Create Power BI dashboard.

Utilise real-world data to identify deeper relationships between customer attributes and purchasing behaviour.

Expand the analysis to include additional features to improve predictions.

## References

Goyal, S. (2021). Customer Purchases Behaviour Dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/sanyamgoyal401/customer-purchases-behaviour-dataset

## License

Customer-Purchase-Behaviour-Analysis © 2024 by Junha Baek is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).  
You are free to share and adapt this work, provided it is for non-commercial purposes and appropriate credit is given to the original author. Any derived work must be distributed under the same license.

For more details, please see the [full license text](LICENSE).

![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)
