# -*- coding: utf-8 -*-
"""
Created on Mon May  2 23:56:19 2022

@author: khali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 

X,y = fn.dataset_dis('50_Startups.csv')
X = fn.missing_value_computation(X, range_ = (0,2))
X = fn.dummyEncoding(X,column_number=3)

# avoiding dummy variable trap
X = X[:,1:]
#X_train , X_test,y_train ,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

#model building
# regressor = LinearRegression()
# regressor.fit(X_train,y_train)

# y_pred = regressor.predict(X_test)

# to add coefficient of X0 which are equal to ones
X = np.append(arr = np.ones((50,1)).astype(int) ,values = X , axis = 1)


X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS( y,X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS( y,X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS( y,X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS( y,X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS( y,X_opt).fit()
regressor_OLS.summary()

X = fn.auto_OLS_model(X, y,0.05)


















