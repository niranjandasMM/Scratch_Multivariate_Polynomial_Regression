from sklearn.metrics import r2_score 
from data import x_train, x_test, y_train, y_test

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Define the degree of the polynomial you want to fit
degree = 2  # You can choose any positive integer value for the degree

# Create a polynomial feature transformer
poly_features = PolynomialFeatures(degree=degree)

# Create a linear regression model
linear_regression = LinearRegression()

# Create a pipeline that applies polynomial transformation and then fits the linear regression
model = make_pipeline(poly_features, linear_regression)

# Fit the model to your training data
model.fit(x_train, y_train)

# Make predictions on the training data
y_pred_train = model.predict(x_train)

# Make predictions on the test data
y_pred_test = model.predict(x_test)

# Calculate the R-squared score for the training data
r2_train = r2_score(y_train, y_pred_train)

# Calculate the R-squared score for the test data
r2_test = r2_score(y_test, y_pred_test)

print("R-squared score (Training data):", r2_train)
print("R-squared score (Test data):", r2_test)

