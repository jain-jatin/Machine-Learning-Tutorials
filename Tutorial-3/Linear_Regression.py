# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import numpy as np;

def Normal_Equation(x, y):
  g=np.dot(np.transpose(x),x)
  h=np.linalg.inv(g)
  z=np.dot(h,np.transpose(x))
  return np.dot(z,y)
  
def differentiate(x,y,t):
  u=np.dot(x,np.transpose(t))
  v=u-y
  v=v.reshape((np.shape(v)[0],1))
  e=np.multiply(v,x)
  q=np.sum(e,axis=0)/len(x)
  q=[round(i,8) for i in q]
  return np.array(q)
  
def grad_descent(x,y,theta):
  d = differentiate(x,y,theta)
  while(np.any(d)!=0):
    theta -= 0.01*d
    d=differentiate(x,y,theta)
  return theta
  
def labelencoder(data1):
  le = LabelEncoder()
  for col in data1.columns:
    if data1[col].dtype == 'object':
      le.fit(data1[col])
      data1[col] = le.transform(data1[col])
  return data1

data = pd.read_csv('Book.csv', index_col=None);
data = labelencoder(data)
X = data['X']
Y = data['Y'];
X = X.values
Y = Y.values

m=len(X);
x1 = [[1 ,q1] for q1 in X]
thetaz = np.zeros(len(x1[0]));
theta1 = Normal_Equation(x1,Y)
print("Normal Equation",theta1);
theta2 = grad_descent(x1,Y,thetaz)
print("gradient Descent",theta2);