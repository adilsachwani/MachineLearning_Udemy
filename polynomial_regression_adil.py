#importing the libraries
import numpy as np #for mathamatucs usage
import matplotlib.pyplot as plt #for plotting purpose
import pandas as pd #for importing and managing datasets

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualizing the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x) ,color='blue')
plt.title('Truth or Bluff - Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)) ,color='blue')
plt.title('Truth or Bluff - Polynomial Regression')
plt.xlabel('Position Level') 
plt.ylabel('Salary')
plt.show()