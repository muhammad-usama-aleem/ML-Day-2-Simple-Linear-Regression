# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test results
y_predict = regressor.predict(x_test)


# visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Salary vs experience(training)')
plt.xlabel('expereince')
plt.ylabel('salary')
plt.show()


# visualizing the training set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Salary vs experience(test)')
plt.xlabel('expereince')
plt.ylabel('salary')
plt.show()
