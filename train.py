import numpy as np
import pandas as pd
import sys

from sklearn.metrics import mean_squared_error

class PolynomialRegression:
    def __init__(self, degrees):
        self.degrees = degrees
        self.w = 0
        self.b = 0

    def gradients(self, X, X_transformed,  y, y_pred, lr):
        m = X.shape[0]  ##(1000,3)
        error = y_pred - y
        # print(f" shape is : {X.shape, self.w.shape, error.shape}") ## ((3, 3), (6, 1), (3, 1))
        
        dw = (1/m) * np.dot(X_transformed.T, error)
        db = (1/m) * np.sum(error)

        self.w -= lr * dw
        self.b -= lr * db

    def predict(self, X, test=False):
        if test:
            x = self.x_transform(X)
            return np.dot(x, self.w) + self.b
        else:
            return np.dot(X, self.w) + self.b

    def train(self, X, y, epochs, lr):
        x = self.x_transform(X)
        m, n = x.shape  ## no. of samples, no. of features
        
        self.w = np.zeros((n, 1))  
        self.b = 0
        losses = []
        
        for epoch in range(epochs):
            # y_pred = w1*x1 + w2*x1^2 + w3*x2 + w4*x2^2 + w5*x3 + w6*x3^2 + b
            y_pred = self.predict(x)
            self.gradients(X, x, y, y_pred, lr)
            # print(f"self.w and self.b are : {self.w, self.b}")

            loss = mean_squared_error(y, y_pred) 

            if epoch % 10 == 0:
                sys.stdout.write(
                    "\n" +
                    "I:" + str(epoch) +
                    " Train-Err:" + str(loss / float(len(X)))[0:5] + 
                    # " Y_pred : " + str(y_pred[0:5]) +
                    # " Y_train : " + str(y_train[0:5]) +
                    "\n"
                )

            losses.append(loss)

    def x_transform(self, X):
        t = X.copy()
        for i in self.degrees:
            X = np.append(X, t**i, axis=1)
        return X

