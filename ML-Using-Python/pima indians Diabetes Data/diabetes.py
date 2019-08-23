# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:52:59 2019

@author: ravip
"""
# This is Binary Classification Model Problem...
import pandas as pd
import numpy as np
# Loadind csv data
data = pd.read_csv('diabetes.csv')

#Check loaded data
data.head()

# gives information about the data types,columns, null value counts, memory usage etc
data.info(verbose=True)

'''
    count tells us the number of NoN-empty rows in a feature.
    mean tells us the mean value of that feature.
    std tells us the Standard Deviation Value of that feature.
    min tells us the minimum value of that feature.
    25%, 50%, and 75% are the percentile/quartile of each features. This quartile information helps us to detect Outliers.
    max tells us the maximum value of that feature.
'''
des = data.describe()
desT = data.describe().T

#below columns cannot have 0 as value.
'''
    Glucose
    BloodPressure
    SkinThickness
    Insulin
    BMI
'''
#hence we will replace 0 value with NaN and then replace with average value.
data_copy = data.copy(deep = True)

data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Check count of null values
null_count = data_copy.isnull().sum()

data.hist(figsize=(10,10))
#by looking at histograms, we will replace null values
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)

null_count_after = data_copy.isnull().sum()

data_copy.hist(figsize =(10,10))

data.shape
#data type analysis - below not working
import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sb.countplot(x=data.dtypes, data = data)
plt.xlabel('Count')
plt.ylabel('datatype')
plt.show()

#checking class distribution
#The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patients
sb.countplot(x='Outcome',data = data)

sb.pairplot(data_copy,hue='Outcome')

corelation = data.corr()
#lets draw heatmap for corelation

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sb.heatmap(data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

data_copy=data_copy.drop(['Outcome'],axis =1)

data_copy.head()

X = pd.DataFrame(sc_x.fit_transform(data_copy),columns =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()

y=data.Outcome

y.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

test_scores =[]
train_scores=[]

for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
#after comparing test and train scores, k=11 gives good results

plt.figure(figsize=(12,5))
p=sb.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p=sb.lineplot(range(1,15),test_scores,marker='o',label='Test Score')

#knn =11 best results
knn=KNeighborsClassifier(11)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sb.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Save Model
import pickle
filename = 'PimaIndModel.sav'
pickle.dump(knn, open(filename,'wb'))

#load model from disk
Loaded_model = pickle.load(open(filename,'rb'))
Loaded_model.score(X_test,y_test)
