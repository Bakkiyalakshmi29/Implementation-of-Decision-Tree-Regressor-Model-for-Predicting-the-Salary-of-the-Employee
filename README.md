# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Bakkiya lakshmi M
RegisterNumber: 212222220006 
*/
import pandas as pd
data=pd.read_csv('/content/Salary_EX7.csv')
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train , y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt, feature_names=x.columns,filled=True)
plt.show()
*/
```

## Output:
HEAD:
![318810687-c5637683-ec58-4738-bf8c-c0598ea5e364](https://github.com/Bakkiyalakshmi29/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119406233/49f7b75e-33fd-4fc7-ad6d-b365e87d432d)

MSE:

![318810969-89af8160-53d6-4274-b3bf-dca17bb9429c](https://github.com/Bakkiyalakshmi29/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119406233/15fadadb-f193-4742-af7c-33b83d0e25c0)

r2:

![318811311-10f5f2e7-c755-49e3-9348-1ac121c737af](https://github.com/Bakkiyalakshmi29/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119406233/a3ffe252-be82-42cd-8386-ab3ff05a4be5)

![318811569-375af8d4-30a0-4cec-8312-2ca3d23e7948](https://github.com/Bakkiyalakshmi29/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119406233/f3290fd3-106f-400a-a6d4-1a324b34bc15)

![318811919-13e3ba1a-0025-4896-959b-6054c445028e](https://github.com/Bakkiyalakshmi29/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119406233/59404d83-8ac2-4e2d-871b-d93364ae5c8a)







## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
