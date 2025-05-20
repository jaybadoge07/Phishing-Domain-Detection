# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:31:24 2022

@author: sc229
"""

import pandas as pd
data=pd.read_csv("C:/sandhyashree/DS/ML&AI/Practice/Logistic Regression/Social_Network_Ads11.csv")
#to find the null value
print(data.isna().sum())
print(data.head())
#from the complete data set selecting column from 3rd to 5th 
X=data.iloc[:,2:5]
#to find correlation between dependent & independent varlible if the output 
#is nearer to 1 it more correlated & viceversa
X_corr=X.corr()
print(X_corr)
X_ind=data.iloc[:,2:4].values
Y=data.iloc[:,-1].values
#checking the shape of the data in X and Y if not same using reshape function 
#it can be reshaped 
print(X_ind.shape)
print(Y.shape)
Y=Y.reshape(Y.shape[0],1)
print(Y.shape)

#Performing Scaling as the salary and age  huge value difference 
from sklearn.preprocessing import StandardScaler
#sc is the object  StandardScalar is the class of sklearn preprocessing module 
sc=StandardScaler()
#X_ind is a varible on which we are performing the scaler 
X_ind=sc.fit_transform(X_ind)
#now the data is scaled 

# sliptting the data into test and training the data
from sklearn.model_selection import train_test_split
#our data is sliptting to 2 set for testing and training
#these are variable these names can be changed it will return 4 variables
X_train,X_test,Y_train,Y_test=train_test_split(X_ind,Y,test_size=0.1,random_state=23)

# importing the classification model 
from sklearn.linear_model import LogisticRegression
#shape of variables should be same
#training the model with data in training part, fit is the function 
log=LogisticRegression()
log.fit(X_train,Y_train)
Y_pred=log.predict(X_test)
#to predict the user input data
print(log.predict([[25,25000]]))
# to find the accuracy of the model
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)


