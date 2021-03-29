# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:16:23 2020

@author: Zhichen
"""

import numpy as np
from scipy.linalg import cholesky,cho_solve
import gc

class Gaussian_Process:
    
    def __init__(self, kernel_='rbf1', l_=1, sigma_f=1, alpha=1, beta=1, gamma=1, sigma_y=1, return_cov=False):
        self.kernel_ = kernel_
        self.l_ = l_
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.return_cov = return_cov
    
    def kernel1(self, x1, x2, l, sigma):
        distance = np.sum(np.square(x1), axis=1).reshape(-1,1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
        K = np.square(sigma) * np.exp(-1 / (2 * np.square(l)) * distance)
        gc.collect()
        return K
        
    def kernel2(self, x1, x2, alpha, beta, gamma):
        distance = np.sum(np.square(x1), axis=1).reshape(-1,1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
        K = alpha * np.exp(-(distance / gamma**2)**beta)
        gc.collect()
        return K
    
    def fit(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        if self.kernel_=='rbf2':
            K = self.kernel2(self.x_train, self.x_train, self.alpha, self.beta, self.gamma) + self.sigma_y * np.eye(len(self.x_train))
            K_s = self.kernel2(self.x_train, x_test, self.alpha, self.beta, self.gamma)
            if self.return_cov==True:
                K_ss = self.kernel2(self.x_test, x_test, self.alpha, self.beta, self.gamma)
        elif self.kernel_=='rbf1':
            K = self.kernel1(self.x_train, self.x_train, self.l_, self.sigma_f) + self.sigma_y * np.eye(len(self.x_train))
            K_s = self.kernel1(self.x_train, x_test, self.l_, self.sigma_f)
            if self.return_cov==True:
                K_ss = self.kernel1(self.x_test, x_test, self.alpha, self.beta, self.gamma)
        
        self.L_ = cholesky(K,lower=True)
        self.K_inv = cho_solve((self.L_,True), self.y_train)
        
        mean_estimate = (K_s.T).dot(self.K_inv)
        if self.return_cov==True:
            var_estimate = K_ss - (K_s.T).dot(np.linalg.inv(self.K)).dot(K_s)
            return mean_estimate,var_estimate
        
        return mean_estimate
