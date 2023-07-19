import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train import PolynomialRegression
from sklearn.metrics import r2_score 
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
# from data import stem_data
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = stem_data()

np.random.seed(42)
X = np.random.rand(1000, 3)
y = 5 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) + np.random.rand(1000)
y = y.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine the features and target into one DataFrame
data = pd.DataFrame(np.hstack((X, y)), columns=['Feature 1', 'Feature 2', 'Feature 3', 'Target'])

# Plot the scatter plot matrix using Seaborn
sns.pairplot(data)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

degree = [2]

model = PolynomialRegression(degrees=degree)
y_train_pred = model.train(X_train, y_train, epochs=100, lr=0.01)

y_test_pred = model.predict(X_test, test=True)

print(f"testing r2 score  is : {r2_score(y_test, y_test_pred)}")

# ## Create the figure
# fig, ax = plt.subplots(figsize=(18, 10))
# # ax.plot(X_train, y_train_pred, label='Predicted (Train)', color='green')
# # ax.plot(X_train, y_train, label='Actual (Train)', color='darkblue')
# ax.plot(X_test, y_test_pred, label='Predicted (Test)', color='green')
# ax.plot(X_test, y_test, label='Actual (Test)', color='red')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Predicted vs Actual Stock Prices')
# ax.legend()
# plt.show()



