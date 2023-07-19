import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def stem_data():
# Read the dataset
    data = pd.read_csv('Levels_Fyi_Salary_Data.csv')
    data = data[['yearsatcompany', 'yearsofexperience', 'totalyearlycompensation', 'basesalary']]

    features = data[['yearsatcompany', 'yearsofexperience', 'basesalary']]
    target = data[['totalyearlycompensation']]

    # Perform min-max scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

def boston_data():
    # Load the Boston Housing dataset
    boston = fetch_california_housing()

    # Create a DataFrame from the dataset
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['target'] = boston.target

    # Perform EDA on the dataset
    print(data.head())  # Display the first few rows of the dataset
    print(data.describe())  # Generate summary statistics of the dataset

    # Visualize the correlation matrix
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Select features and target variable
    features = ['RM', 'LSTAT']  # Choose two features for polynomial regression
    target = 'target'

    # Create scatter plots of the selected features against the target variable
    for feature in features:
        plt.scatter(data[feature], data[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f"Scatter plot: {feature} vs {target}")
        plt.show()

boston_data()