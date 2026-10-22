# Titanic Dataset - Data Cleaning and Preprocessing

This repository contains the Python script used for acquiring, cleaning, and preprocessing the publicly available Titanic dataset.

## Objective
To demonstrate handling missing values, identifying outliers, and performing feature encoding to prepare raw data for machine learning models.

## Tools Used
- Python
- Pandas
- NumPy
- Seaborn

## Process Highlights
- Handled missing values using median (numerical) and mode (categorical) imputation.
- Managed extreme outliers in the 'fare' column using the Interquartile Range (IQR) Winsorization technique.
- Transformed categorical variables into numerical format using One-Hot Encoding.
- Removed redundant features to prevent multicollinearity.
