# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:57:50 2019

@author: ravip
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('winequality-red.csv')

data.head()

data.info()

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='fixed acidity', data = data)
#Here we see that fixed acidity does not give any specification to classify the quality.

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='volatile acidity', data = data)
#As volatile acidity goes down, quality improves


fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='citric acid', data = data)
#As citric acid goes up, quality improves

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='residual sugar', data = data)
#does not affect

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='chlorides', data = data)
#chlorides reduces, improves quality

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='free sulfur dioxide', data = data)
#does not affect

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='total sulfur dioxide', data = data)
#does not affect

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='sulphates', data = data)
#As sulphates goes up, quality improves

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='alcohol', data = data)
#As alcohal goes up, quality improves

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='density', data = data)
#does not affect

fig=plt.figure(figsize=(10,6))
sns.barplot(x='quality',y='pH', data = data)
#does not affect

#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2,6.5,8)
group_name = ['bad','good']
data['quality']=pd.cut(data['quality'],bins = bins,labels = group_name)

from sklearn import preprocessing

label_quality = preprocessing.LabelEncoder()

data['quality']=label_quality.fit_transform(data['quality'])

import seaborn as sns

sns.countplot(data['quality'])

y=data['quality']
X=data.drop('quality',axis = 1)

X=X.drop(['pH','density','total sulfur dioxide','free sulfur dioxide','residual sugar','fixed acidity'],axis = 1)

sc = preprocessing.StandardScaler()

X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

