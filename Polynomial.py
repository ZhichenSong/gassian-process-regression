# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:05:52 2020

@author: Zhichen
"""

import numpy as np

class PolynomialRegression:
    
    def __init__(self,p=3):
        self.coef_ = 0
        self.intercept_ = 0
        self.p = p
    
    def poly_transform(self,X_train):
        flag = 1
        X_train_copy = X_train.copy()
        while flag<self.p:
            for i in range(len(X_train[0])):
                X_poly = np.hstack([X_train_copy,(X_train[:,i]**(flag+1)).reshape(-1,1)])
                X_train_copy = X_poly
            flag += 1
        return X_poly
    
    def fit(self,X_train,y_train):
        X_poly = self.poly_transform(X_train)
        X_b = np.hstack([np.ones((len(X_poly),1)),X_poly])
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def predict(self,X_test):
        X_b = self.poly_transform(X_test)
        X_b = np.hstack([np.ones((len(X_b),1)),X_b])
        y_predict = X_b.dot(self.theta)
        for i in range(len(y_predict)):
            if y_predict[i]<0:
                y_predict[i] = 0
        return y_predict
