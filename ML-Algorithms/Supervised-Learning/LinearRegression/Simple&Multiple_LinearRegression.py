# Simple or Multivariate Linear Regression 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lm

data = pd.read_csv('hiring.csv')

#replace NaN values

data.test_score = data.test_score.fillna(data.test_score.median()) 
data.experience = data.experience.fillna('zero')

#Convert numbers in words to number
from word2number import w2n 
data.experience = data.experience.apply(w2n.word_to_num)

corelation = data.corr()

Lmodel = lm.LinearRegression()

X=data[['experience','test_score','interview_score']]
y=data[['salary']]

Lmodel.fit(X,y)

coef = Lmodel.coef_
intercept = Lmodel.intercept_

Lmodel.predict([[2,9,6]])

#Saving Model-
import pickle
pickle.dump(Lmodel,open('LR_Model.pkl','wb'))
