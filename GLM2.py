# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:05:52 2020

@author: Zhichen
"""

import numpy as np

class LinearRegression:
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self,x_train,y_train):
        if self.fit_intercept==True:
            x = np.hstack([np.ones((len(x_train),1)),x_train])
            self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_train)
            self.intercept_ = self.theta[0]
            self.coef_ = self.theta[1:]
        else:
            x = x_train
            self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_train)
            self.coef_ = self.theta
        return self
    
    def predict(self,x_test):
        x = np.hstack([np.ones((len(x_test),1)),x_test])
        y_predict = x.dot(self.theta)
        return y_predict
