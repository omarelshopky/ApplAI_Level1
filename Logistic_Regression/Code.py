import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

train = pd.read_csv('train_session4.csv')
test = pd.read_csv('test_session4.csv')
train.isnull().sum()

train =  train.drop(['Cabin'],axis=1)
test =  test.drop(['Cabin'],axis=1)
train = train.dropna(axis = 0,subset=['Embarked'],how='any')

# Nearly 20% of data is missing, can't risk losing all of the data
(train.Age.isnull().sum() / train.shape[0]) * 100

# Use mean/median imputation
train['Age'].fillna( train['Age'].median() , inplace=True)
test['Age'].fillna( train['Age'].median() , inplace=True)
test['Fare'].fillna( train['Fare'].median() , inplace=True)
# No missing values now
train.info()

# drop those who don't have logical impact eg : names, id's...
# Sex
train["Sex"]= np.where(train["Sex"]=="female",0,1)
test["Sex"]= np.where(test["Sex"]=="female",0,1)

# Name
train =  train.drop(['Name'],axis=1)
test =  test.drop(['Name'],axis=1)

# Ticket Number
train =  train.drop(['Ticket'],axis=1)
test =  test.drop(['Ticket'],axis=1)

# Id
ID = test.PassengerId
train =  train.drop(['PassengerId'],axis=1)
test =  test.drop(['PassengerId'],axis=1)

# Embarked
train["Embarked"]= np.where(train["Embarked"]=="C",1,np.where(train["Embarked"]=="S",2,3))
test["Embarked"]= np.where(test["Embarked"]=="C",1,np.where(test["Embarked"]=="S",2,3))

train.head()

X_train =  train.drop(['Survived'],axis=1)
y_train = train.Survived
X_test = test

def sigmoid(input):    
  output = 1 / (1 + np.exp(-input))
  return output
  
def costFunction(theta, X, y):
    m = y.size  
    J = 0
    h = sigmoid(np.dot(X, theta.T))
    J = 1/m * np.sum(-y*np.log(h) - (1-y) * np.log(1-h))
    return J
    
def gradientDecsent(iterations, alpha, x, y):
  m = x.shape[0]
  x = np.concatenate([np.ones((m, 1)), x], axis=1)
  y = np.array(y)
  y = np.reshape(y, (len(y),1))  
  theta = np.full((1, x.shape[1]), 0)
  # theta = theta.copy()
  cost_history = []
  for i in range(iterations):
    theta = theta - (alpha / m) * np.transpose(sigmoid(np.dot(x, theta.T)) - y).dot(x)
    cost_history.append(costFunction(theta, x, y))
  return theta, cost_history 

theta, cost_history = gradientDecsent(200000, 0.002, X_train, y_train)
print(theta)

plt.plot(np.arange(len(cost_history)), cost_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

x_ones = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)
y_pred = sigmoid(np.dot(x_ones, theta.T))

y_pred_final = []
for i in y_pred:
  if i >= 0.5:
    y_pred_final.append(1)
  else:
    y_pred_final.append(0)
print(y_pred_final)   

#saving results
# y_pred = np.array(y_pred)
submission = {'PassengerId':ID,'Survived':y_pred_final}
submission = pd.DataFrame(submission)
submission.to_csv("Submission.csv", index=False)
