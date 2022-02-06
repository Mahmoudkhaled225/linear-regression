"""
THE SPARKS FOUNDATION
DATA SCIENCE and BUSINESS ANALYTICS INTERNSHIP (CURRENT PATCH)
Task 1 --> Prediction of student grades depend on studied hour
Note
i did all the analysis phase and i got my insights and started perform the algorithm
"""

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)

# independent variable
X = dataset.iloc[:, :-1].values
# independent variable
y = dataset.iloc[:, 1].values

# data splitting for training and testing 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

linearReg = LinearRegression()
linearReg.fit(X_train, y_train)

# predicted values
y_pred = linearReg.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, linearReg.predict(X_train), color="blue")
plt.title("Grades And Hours Studied (Training set)")
plt.xlabel("Hours Studied")
plt.ylabel("Grades")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, linearReg.predict(X_train), color="blue")
plt.title("Grades And Hours Studied (Testing set)")
plt.xlabel("Hours Studied")
plt.ylabel("Grades")
plt.show()
