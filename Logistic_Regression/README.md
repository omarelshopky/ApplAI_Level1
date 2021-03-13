# Logistic Regression Algorithm
Implement the Logistic Regression Algorithm to predict survivors of shipwreck using Titanic Data.

Having [Training Data](https://github.com/omarhesham2/ApplAI_Level1/blob/main/Logistic_Regression/train.csv) and [Test Data](https://github.com/omarhesham2/ApplAI_Level1/blob/main/Logistic_Regression/test.csv)

## Code :

### Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Initialize & Organize Data
```python
train = pd.read_csv('train_session4.csv')
test = pd.read_csv('test_session4.csv')
train.isnull().sum()
```

```python
train =  train.drop(['Cabin'],axis=1)
test =  test.drop(['Cabin'],axis=1)
train = train.dropna(axis = 0,subset=['Embarked'],how='any')
```

### Nearly 20% of data is missing, can't risk losing all of the data
```python
(train.Age.isnull().sum() / train.shape[0]) * 100
```

### Use mean/median imputation
```python
train['Age'].fillna( train['Age'].median() , inplace=True)
test['Age'].fillna( train['Age'].median() , inplace=True)
test['Fare'].fillna( train['Fare'].median() , inplace=True)
```

### No missing values now
```python
train.info()
```

### drop those who don't have logical impact eg : names, id's...
#### Sex
```python
train["Sex"]= np.where(train["Sex"]=="female",0,1)
test["Sex"]= np.where(test["Sex"]=="female",0,1)
```
#### Name
```python
train =  train.drop(['Name'],axis=1)
test =  test.drop(['Name'],axis=1)
```
#### Ticket Number
```python
train =  train.drop(['Ticket'],axis=1)
```
#### Id
```python
ID = test.PassengerId
train =  train.drop(['PassengerId'],axis=1)
test =  test.drop(['PassengerId'],axis=1)
```
#### Embarked
```python
train["Embarked"]= np.where(train["Embarked"]=="C",1,np.where(train["Embarked"]=="S",2,3))
test["Embarked"]= np.where(test["Embarked"]=="C",1,np.where(test["Embarked"]=="S",2,3))
```
```python
test =  test.drop(['Ticket'],axis=1)
train.head()
```

```python
X_train =  train.drop(['Survived'],axis=1)
y_train = train.Survived
X_test = test
```

### Sigmoid Function
```python
def sigmoid(input):    
  output = 1 / (1 + np.exp(-input))
  return output
```

### Cost Function
```python
def costFunction(theta, X, y):
    m = y.size  
    J = 0
    h = sigmoid(np.dot(X, theta.T))
    J = 1/m * np.sum(-y*np.log(h) - (1-y) * np.log(1-h))
    return J
```

### Gradient Decsent Function
```python
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
```

```python
#theta, cost_history = gradientDecsent(200000, 0.0042, x, y)
theta, cost_history = gradientDecsent(200000, 0.002, X_train, y_train)
print(theta)
```

```python
plt.plot(np.arange(len(cost_history)), cost_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J') 
```

```python
x_ones = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)
y_pred = sigmoid(np.dot(x_ones, theta.T))
```

### saving results
```python
submission = {'PassengerId':ID,'Survived':y_pred_final}
submission = pd.DataFrame(submission)
submission.to_csv("Submission.csv", index=False)
```

