import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set

# Shoe sizes for all ages(years)
x_train = [[3], [7], [10], [15], [18]]
# Age groups in relation to sizes
y_train = [[2], [4], [5], [6], [8]]

# Testing set

# Shoe sizes for all ages(years)
x_test = [[6], [8], [13], [23]]
# Age groups in relation to sizes
y_test = [[3.5], [4.5], [5.5], [9]]

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
x = np.linspace(0, 25, 100)
y = regressor.predict(x.reshape(x.shape[0], 1))
plt.plot(x, y)

# Set the degree of the Polynomial Regression model
quadratic_feature = PolynomialFeatures(degree=3)

# Takes our quadratic equation in line with our declared
# degrees and transforms it for use
x_train_quadratic = quadratic_feature.fit_transform(x_train)
x_test_quadratic = quadratic_feature.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_feature.transform(x.reshape(x.shape[0], 1))

# Plots our graph with a solid and curved line
plt.plot(x, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title("Age groups(in years) vs shoe sizes")
plt.xlabel("Shoe sizes")
plt.ylabel("Age group(in years)")
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

# References
# https://www.geeksforgeeks.org/numpy-linspace-python/
# https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
