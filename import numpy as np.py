import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
# data = pd.read_csv('your_dataset.csv')
# X = data[['Experience']].values  # Replace with your actual feature column
# y = data['Salary'].values  # Replace with your actual target column

# Example dummy data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply polynomial transformation
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)

# Train the model
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Make predictions
X_test_poly = poly_reg.transform(X_test)
y_pred = poly_model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plotting
plt.scatter(X, y, color='blue')
plt.plot(X, poly_model.predict(poly_reg.fit_transform(X)), color='red')
plt.title('Polynomial Regression - Salary Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
