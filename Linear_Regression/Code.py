# Import the needed libraries

import numpy as np                          # Dealing with Numpy Arrays
import pandas as pd                         # Dealing with Dataframes
import matplotlib.pyplot as plt             # matplotlib is used for ploting graphs
from IPython.display import HTML
from sklearn.preprocessing import MinMaxScaler

# Lets the plots appears in the notebook
%matplotlib inline
# =====================================================================
  
# Initialize our train-test Data
Training_Data = pd.read_csv('Training Data - Training Data.csv')
Testing_Data = pd.read_csv('Testing Data - Testing Data.csv')


Training_Data.head(8) #quick look at first 8 records in my training data
# =====================================================================

# The data is having some null values, so we are going to replace them with the mean of the other values in the same column
Training_Data.isnull().sum()
Training_Data.total_bedrooms = Training_Data.total_bedrooms.fillna(Training_Data.total_bedrooms.mean())
Training_Data.isnull().sum()
  
Training_Data.corr()
# =====================================================================

# Drop some not important data
Training_Data.drop(['HouseID'], axis = 1, inplace = True) #dropping the id column
Testing_Data.drop(['HouseID'], axis = 1, inplace = True)

## here we make the range of all numbers between -1 to 1 to avoid overflow and go faster to the global minimum 
scaler = MinMaxScaler()
Training_Data.iloc[:,0:-1] = pd.DataFrame(scaler.fit_transform(Training_Data.iloc[:,0:-1]), columns=Training_Data.columns[0:-1])
Testing_Data = pd.DataFrame(scaler.fit_transform(Testing_Data), columns=Testing_Data.columns)

# =====================================================================

# Separate the Training_Data to x_train and y_train, the Testing_Data as x_test nad converting them to numpy arrays
x_train = np.array(Training_Data.iloc[:,0:len(Training_Data.iloc[0,:]) - 1])
y_train = np.array(Training_Data.iloc[:,-1])
y_train = y_train.reshape(y_train.shape[0], 1)


x_test = np.array(Testing_Data)

x_train = np.append(np.ones((x_train.shape[0], 1)), x_train, axis = 1)
x_test = np.append(np.ones((x_test.shape[0], 1)), x_test, axis = 1)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print()
# =====================================================================

# Define alpha, number of iterations using in gradient descent and m (number of training examples)
alpha = 0.001
num_of_iterations = 1000
m = int(x_train.shape[0])
m
# =====================================================================

# Implement a hypothesis function which creates a line that we will use to predict any house price using it's size an input
def Predictive_Line(X, Theta):
    Predictions = None
    Predictions = np.dot(X, Theta)
    return Predictions
# =============================================================

# Cost Function Implementation
def Calculate_Cost(X, Theta, Y, m):
    J = 0
    P = Predictive_Line(X, Theta)
    J = ((1/(2 * m)) * np.sum(np.square(P - Y), axis = 0))
    return J
# =============================================================

# Gradient Descent Implementation
def Gradient_Descent(X, Y, Theta, alpha, num_iters, m):
  for i in range(num_iters):
    P = Predictive_Line(X, Theta)
    Theta = Theta - (alpha / m) * (((P - Y).T).dot(X)).T
  return Theta
# =============================================================

# Generating a column of predictions for x_test and calculation the cost.
Theta = []
for i in range(0, x_train.shape[1]):
  Theta.append(0)

Theta = np.array(Theta).reshape((x_train.shape[1], 1)) # shape of theta is (n,1)

Theta = Gradient_Descent(x_train, y_train, Theta, alpha, num_of_iterations, m)

Cost = Calculate_Cost(x_train, Theta, y_train, m)

y_test = Predictive_Line(x_test, Theta)
print('Theta = ')
print(str(Theta))
print()
print('Cost = '+str(Cost))
# =============================================================

# Replacing house_value values in sample submission file with y_test values and making a new submission file.
Submission = pd.read_csv('sampleSubmission - sampleSubmission.csv')
Submission.iloc[:,1] = y_test
Submission.house_value = Submission.house_value.fillna(Submission.house_value.mean())
Submission.to_csv('Submission File.csv', index = False)
Submission.isnull().sum()
