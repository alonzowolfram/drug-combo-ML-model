#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:52:43 2021

@author: alonzowolf
"""

import argparse
import sys
import random
import warnings
import pandas                            
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                  
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from scipy.stats import spearmanr # Spearman's correlation coefficient function.
import statistics as stats
from datetime import datetime
from datetime import date
import pickle
xgb.set_config(verbosity=2)

# Load dataset.
dataset_filename = '/Users/alonzowolf/Dropbox/Work/Thesis_projects/Common_material/Data/Processed_data/Machine_learning/test.csv'
target_column = 2
grouping_column = 0
pred_start_column = 3
pred_end_column = -99
data = pandas.read_csv(dataset_filename)
folds = 5
predstartcol = pred_start_column
predendcol = pred_end_column 
targetcol = target_column 
groupscol = grouping_column                     
if predendcol==-99:
    X = data.iloc[:,predstartcol:len(data.columns)]
else:
    X = data.iloc[:,predstartcol:predendcol]
y = data.iloc[:,targetcol]
groups = data.iloc[:,groupscol]

# Scale predictor values.
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X_scaled = scaling.transform(X)
X_scaled = pandas.DataFrame(X_scaled) # This part necessary because scaling transforms it into a numpy ndarray, and later functions require a Pandas DataFrame as input. 

# Split into training and test. 
test_train_ratio = 0.2
dataset_split = list()
unique_groups = list(set(groups)) # All the (unique) groups. The parameter groups is not unique. 
n_groups = len(unique_groups) # The total number of groups at the start. 
test_size = int(n_groups * test_train_ratio) # Size of the test set, in number of groups. 
    
test_set = list() # List containing the indices of the rows to be used in the training set.
    
# Randomly select m groups for this fold, where m = test_size (the number of groups in the test split.)
random.seed(1026)
m_groups = random.sample(unique_groups, test_size) # The unique groups themselves, not their indices!
    
# Populate fold i with the indices of all rows in the data set belonging to the groups in m_groups.
for j in m_groups:
    # Get all the rows belonging to group j.
    indices = [i for i, x in enumerate(groups) if x == j]
    test_set.extend(indices)
            
    # Remove group j from the list of unique groups so they won't be used for subsequent folds. 
    unique_groups.remove(j)
        
dataset_split.append(test_set)
        
all_indices = set(range(len(X_scaled))) 
test_indices = dataset_split[0]
train_indices = list(all_indices.difference(test_indices))
X_scaled_train = X_scaled.iloc[train_indices,:]
y_train = y.iloc[train_indices] # y.iloc[train_indices,:]
X_scaled_test = X_scaled.iloc[test_indices,:]
y_test = y.iloc[test_indices] # y.iloc[test_indices,:]
groups_train = groups[train_indices]
groups_test = groups[test_indices]

# Define models.
models = dict()
n_trees = 5
models['RandomForest'] = RandomForestRegressor(n_estimators=n_trees, verbose = 11) # Verbose

# Evaluate the model on the training set.
# Perform cross-validation.
results = dict()
results_raw = pandas.DataFrame(index = list(models.keys()), columns = list(range(1, folds+1)))
results_raw_by_group = pandas.DataFrame(index = list(models.keys()), columns = list(set(groups)))
    
items = list(models.items())
i = 0
name, model = items[i] # items[i] instead of random.sample(items, 1)[0] for grid search.
    
# Evaluate the model.
# Split training/test. 
training_split = list()

unique_groups = list(set(groups_train)) # All the (unique) groups. The parameter groups is not unique. 
n_groups = len(unique_groups) # The total number of groups at the start. 
fold_size = int(n_groups / folds) # Size of each fold, in number of groups. 

for i in range(folds): # For() loop going over each fold. 
    fold = list() # List containing the indices of the rows to be used in fold i.
        
    # Randomly select m groups for this fold, where m = fold_size (the number of groups in each fold.)
    m_groups = random.sample(unique_groups, fold_size) # The unique groups themselves, not their indices!
        
    # Populate fold i with the indices of all rows in the data set belonging to the groups in m_groups.
    for j in m_groups:
        # Get all the rows belonging to group j.
        indices = [i for i, x in enumerate(groups) if x == j]
        fold.extend(indices)
            
        # Remove group j from the list of unique groups so they won't be used for subsequent folds. 
        unique_groups.remove(j)
        
    training_split.append(fold) # Append() here because we want dataset_split to be a list of lists. Extend() above (fold.extend()) because we want fold to be a simple vector (list) of indices without any kind of separation

split = training_split
    
# Loop through each fold and build the model, make predictions, and then score the model. 
fold_scores = list()
all_indices = set(X_scaled_train.index) # originally set(range(len(X_scaled_train))), but that re-indexes it ... 
group_scores = {}
for i in range(len(split)): 
    test_indices_fold_i = split[i]
    train_indices_fold_i = list(all_indices.difference(test_indices_fold_i))
    X_scaled_train_fold_i = X_scaled_train.loc[train_indices_fold_i,:]
    y_train_fold_i = y_train.loc[train_indices_fold_i] # y.iloc[train_indices,:]
    X_scaled_test_fold_i = X_scaled_train.loc[test_indices_fold_i,:]
    y_test_fold_i = y_train.loc[test_indices_fold_i] # y.iloc[test_indices,:]
        
    # Build model. 
    model.fit(X_scaled_train_fold_i, y_train_fold_i) # Do only scikit-learn models have the .fit() method?
        
    # Use model to predict. 
    predicted = pandas.Series(model.predict(X_scaled_test_fold_i))
        
    # Evaluate model.
    predicted = predicted.to_frame()
    y_test_fold_i = y_test_fold_i.to_frame()
    temp = y_test_fold_i.join(predicted.set_index(y_test_fold_i.index))
    
    group_scores_i = {}
    # For each group, score the performance of the model using the scoring function defined by scoring_function. 
    groups_sub = groups_train[test_indices_fold_i]
    y_true = temp.iloc[:,1]
    y_pred = temp.iloc[:,0]
    unique_groups = list(set(groups_sub))
    for j in unique_groups:
        print(j)
        # Get the indices of y_true, y_pred that correspond to group i. 
        indices =  groups_sub[groups_sub==j].index # [j for j, x in enumerate(groups_sub) if x == i] <- This doesn't work for Pandas series, because it resets the indices. 
        # Subset y_true, y_pred to include only entries in group i. 
        y_true_j = y_true[indices] # Can we do it this way? Yes, if this is a series object of pandas.core.series module. If this is a dataframe, we will need to use .iloc[:,:], of course.
        y_pred_j = y_pred[indices]
        
        # Score and add the score to the scores list.
        group_scores_i[j], p = spearmanr(y_true_j, y_pred_j)
        
    #group_scores_i = score_model(temp.iloc[:,1], temp.iloc[:,0], groups_train[test_indices_fold_i], scoring_function)
        
    # Add mean_score (mean of scores of the individual groups in fold i) to fold_scores.
    fold_scores.append(stats.mean(group_scores_i.values()))
    # Add individual group scores to group_scores.
    group_scores.update(group_scores_i)
    
    #scores = group_scores
    #if scores is not None:
    #    # Store summarized results in results.
    #    results[name] = scores
    #    mean_score, std_score = mean(scores), std(scores)
    #    
    #    print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
    #    print()
    #    
    #    # Store raw results in results_raw (average score over all groups for a given fold) and results_raw_by_group (individual score of each group.)
    #    results_raw.loc[name] = scores
    #    # Making the results_raw_by_group is going to be a little more involved. We have to make sure that that particular cell line was actually in a test set for that model (because of the way cv_split_by_groups works, not all cell lines will end up in a test set.)
    #    for group in results_raw_by_group.columns:
    #        # Check if that group is actually in group_scores.
    #        if group in group_scores:
    #            results_raw_by_group.loc[name, group] = group_scores[group]
    #        else:
    #            results_raw_by_group.loc[name, group] = 'NA'
    #    
    #else:
    #    print('>%s: error' % name)
    #    print()