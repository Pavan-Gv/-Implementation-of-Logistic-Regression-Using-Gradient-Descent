# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.<br>
2. Read the given dataset and assign x and y array.<br>
3. Split x and y into training and test set.<br>
4. Scale the x variables.<br>
5. Fit the logistic regression for the training set to predict y.<br>
6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity<br>
7. Plot the training set results.<br>

Program:
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: G Venkata Pavan Kumar
RegisterNumber: 212221240013
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Social_Network_Ads (1).csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("red","black")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("gray","white"))(i),label=j)
plt.title("Training Set of Logistic Regression")
plt.xlabel("Age Recorded")
plt.ylabel("EstimatedSalary")
plt.legend()
plt.show()
```

## Output:
![4 1](https://user-images.githubusercontent.com/94827772/173214982-6e14305f-934e-4cb2-abf2-12d23b6f3396.png)

![4 2](https://user-images.githubusercontent.com/94827772/173214980-306d2f14-4352-4979-af37-99576d2d3533.png)

![4 3](https://user-images.githubusercontent.com/94827772/173214978-8c626534-b6c6-4f82-b1a1-b71f0cea8da6.png)

![4 4](https://user-images.githubusercontent.com/94827772/173214976-19e8f165-6ac6-480e-9c36-8dc9874c9757.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

