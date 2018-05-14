#importing the libraries
import numpy as np #for mathamatucs usage
import matplotlib.pyplot as plt #for plotting purpose
import pandas as pd #for importing and managing datasets


#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values


#splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""


#fitting simple linear  model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting the test set results
y_pred = regressor.predict(x_test)


#visulaize the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vx Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('Salary vx Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()