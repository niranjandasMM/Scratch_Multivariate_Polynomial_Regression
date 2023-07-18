import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
  
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
