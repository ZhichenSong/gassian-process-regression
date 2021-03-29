# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:11:04 2020

@author: Zhichen
"""

import numpy as np
import random as rd
from random import randrange

class RandomForest:
    
    def __init__(self,n_trees=5,sub_sample_ratio=0.20,sub_feature_ratio=0.63,max_depth=10):
        self.n_trees = n_trees
        self.sub_sample_ratio = sub_sample_ratio
        self.sub_feature_ratio = sub_feature_ratio
        self.max_depth = max_depth
    
    def sub_sample(self,x_train,y_train):
        self.sub_sample_size = x_train.shape[0] * self.sub_sample_ratio
        sub_xtrain = []
        sub_ytrain = []
        while len(sub_xtrain)<self.sub_sample_size:
            index = randrange(len(x_train)-1)
            sub_xtrain.append(x_train[index])
            sub_ytrain.append(y_train[index])
        return sub_xtrain,sub_ytrain
    
    def sub_space(self,x_train,y_train):
        if type(self.sub_feature_ratio)==float:
            self.sub_space_size = len(x_train[0]) * self.sub_feature_ratio
        elif self.sub_feature_ratio=='sqrt':
            self.sub_space_size = np.sqrt(len(x_train[0]))
        elif self.sub_feature_ratio=='log':
            self.sub_space_size = np.log2(len(x_train[0]))
        else:
            raise ValueError('Feature ratio not recognized')
        sub_xtrain_space = []
        try:
            flag = x_train.shape[0]
            del flag
        except:
            x_train = np.array(x_train)
        x_indices = []
        while len(sub_xtrain_space)<self.sub_space_size:
            index = randrange(len(x_train[0])-1)
            x_indices.append(index)
            sub_xtrain_space.append(x_train[:,index])
        self.feature_index = x_indices
        return np.array(sub_xtrain_space).T,np.array(y_train)
    
    def calculate_entropy(self,y):
        entropy = np.std(y)
        return entropy
    
    def find_split(self,x, y):
        """Given a dataset and its target values, this finds the optimal combination
        of feature and split point that gives the maximum information gain."""
        # Need the starting entropy so we can measure improvement...
        start_entropy = self.calculate_entropy(y)
        # Best thus far, initialised to a dud that will be replaced immediately...
        best = {'infogain' : -np.inf}
        # Loop every possible split of every dimension...
        for i in range(x.shape[1]):
            sorted_x = np.sort(np.unique(x[:,i]))
            for j in range(len(sorted_x)):
                split = sorted_x[j]
                # left branch entropy
                left_indices = np.array(np.where(x[:,i]<=split))
                y_left = [y[ind] for ind in left_indices][0]
                p_left = len(y_left)/len(y)
                entropy_left = self.calculate_entropy(y_left)
                # right branch entropy
                right_indices = np.array(np.where(x[:,i]>split))
                y_right = [y[ind] for ind in right_indices][0]
                p_right = len(y_right)/len(y)
                entropy_right = self.calculate_entropy(y_right)
                # information gain for the split
                infogain = start_entropy - (p_left*entropy_left + p_right*entropy_right)
                if infogain > best['infogain']:
                    best = {'feature' : self.feature_index[i],
                            'split' : split,
                            'infogain' : infogain, 
                            'left_indices' : left_indices[0],
                            'right_indices' : right_indices[0]}
        return best
    
    def build_tree(self, x, y, max_depth=10):
        """Check if either of the stopping conditions have been reached. If so generate a leaf node..."""
        if max_depth==1 or (y==y[0]).all():
            # Generate a leaf node...
            return {'leaf' : True, 'value' : np.mean(y)+rd.uniform(-0.5,0.5)}
        else:
            x_subspace,y_subspace = self.sub_space(x,y)
            move = self.find_split(x_subspace, y_subspace)
            try:
                left = self.build_tree(x[move['left_indices'],:], y[move['left_indices']], max_depth - 1)
                right = self.build_tree(x[move['right_indices'],:], y[move['right_indices']], max_depth - 1)
                
                return {'leaf' : False,
                        'feature' : move['feature'],
                        'split' : move['split'],
                        'infogain' : move['infogain'],
                        'left' : left,
                        'right' : right}
            except:
                return {'leaf' : True, 'value' : np.mean(y_subspace)}
            
    def fit(self,x_train,y_train):
        ensemble = []
        n_tree = 0
        while n_tree<self.n_trees:
            sub_xtrain,sub_ytrain = self.sub_sample(np.array(x_train),np.array(y_train))
            tr = self.build_tree(np.array(sub_xtrain),np.array(sub_ytrain),max_depth=self.max_depth)
            ensemble.append(tr)
            n_tree+=1
        
        self.ensemble = ensemble
        return self
    
    def predict_one(self,tree,samples):
        """Predicts class for every entry of a data matrix."""
        ret = np.empty(samples.shape[0], dtype=float)
        ret.fill(-1)
        indices = np.arange(samples.shape[0])
        def tranverse(node, indices):
            nonlocal samples
            nonlocal ret
            if node['leaf']:
                ret[indices] = float(node['value'])
            else:
                going_left = samples[indices, node['feature']] <= node['split']
                left_indices = indices[going_left]
                right_indices = indices[np.logical_not(going_left)]
                
                if left_indices.shape[0] > 0:
                    tranverse(node['left'], left_indices)
                    
                if right_indices.shape[0] > 0:
                    tranverse(node['right'], right_indices)
        tranverse(tree, indices)
        return ret
    
    def predict(self,x_test):
        predictions = np.array([self.predict_one(tree, x_test) for tree in self.ensemble])
        y_pred_final = np.mean(predictions,axis=0)
        return y_pred_final
