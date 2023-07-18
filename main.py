import numpy as np
import matplotlib.pyplot as plt
from train import PolynomialRegression, r2_score_model
from sklearn.metrics import r2_score 
from data import x_train, x_test, y_train, y_test

# np.random.seed(42)
# # X = np.random.rand(1000, 3)
# X = np.array( [[1,2,3], [4,5,6], [7,8,9]] )
# y = 5 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) + np.random.rand(3)
# y = y.reshape(-1, 1)

degree = [2] ## if you need degree 3 then = [2,3], if 4 then = [2,3,4] and so on ....

model = PolynomialRegression(degrees=degree)
model.train(x_train, y_train, epochs=10, lr=0.0001)

# plt.plot(l, 'r-') ; plt.show()

# # # # Plotting
## Plotting in Progress

y_test_pred = model.predict(x_test, test=True)
print( "r2 score model is : ",  r2_score_model(y_test, y_test_pred ) )
print(f"r2 score skelarn is : {r2_score(y_test, y_test_pred)}")



