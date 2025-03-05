# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas for data handling, numpy for numerical operations, matplotlib.pyplot for visualization, and sklearn for machine learning functions. These libraries help in data preprocessing, model training, evaluation, and visualization.
2. Read student_scores.csv and extract features (X = Hours studied) and target (Y = Scores). Split the dataset into training and testing sets using train_test_split() to ensure the model is trained and tested on separate data.
3. Create an instance of LinearRegression and fit it to the training data (X_train, Y_train). This step enables the model to learn the relationship between study hours and scores.
4. Use the trained model to predict scores for X_test and compare them with actual values. Compute error metrics such as MSE, MAE, and RMSE to measure the model's accuracy. 
5.Plot the training and test data along with the regression line to observe the model's performance. Analyze the plots and error metrics to understand how well the model predicts student scores.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Simon Malachi S
RegisterNumber:  212224040318
*/
```
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("House vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,Y_pred,color="blue")
plt.title("House vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
head values

![Screenshot 2025-03-04 094310](https://github.com/user-attachments/assets/ec2365dd-ede5-4e30-9d1c-4271ddcc210a)

tail values

![Screenshot 2025-03-04 094316](https://github.com/user-attachments/assets/1ded3eff-9833-47d1-8fe0-c5102f3599f8)

X values

![Screenshot 2025-03-04 094433](https://github.com/user-attachments/assets/b03c9660-760e-486d-8314-ba238886879b)

Y values

![Screenshot 2025-03-04 094530](https://github.com/user-attachments/assets/ee6ec4f4-f70a-4b8f-9edb-d3512ef6a64b)

Predicted values

![Screenshot 2025-03-04 094802](https://github.com/user-attachments/assets/d2abd067-57bd-4737-aefe-f0be4bb1551f)

Actual values

![Screenshot 2025-03-04 094716](https://github.com/user-attachments/assets/3ccfa301-d935-4383-ac3f-f7080b852876)

Testing set

![Screenshot 2025-03-05 084508](https://github.com/user-attachments/assets/35a14fee-b41f-4b7f-abc0-a4d40190c92b)

Training set
![Screenshot 2025-03-05 084439](https://github.com/user-attachments/assets/0a62d12d-11a0-4af1-a85c-1438ccfc6af4)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
