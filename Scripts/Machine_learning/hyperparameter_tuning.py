# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:13:01 2021

@author: lwfong

Tutorials followed:
    https://kevinvecmanis.io/machine%20learning/hyperparameter%20tuning/dataviz/python/2019/05/11/XGBoost-Tuning-Visual-Guide.html#learning_rate (XGBoost)
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 (RandomForest)
    https://www.geeksforgeeks.org/grid-searching-from-scratch-using-python/
Implementing random search from scratch: (RandomForest)
    https://stackoverflow.com/questions/61818704/building-a-custom-randomsearchcv-using-python
Why random search is just as good as grid: 
    https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    https://stats.stackexchange.com/a/209409
Etc:
    https://pierpaolo28.github.io/blog/blog25/
    https://www.oreilly.com/library/view/evaluating-machine-learning/9781492048756/ch04.html
    https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b
    https://stackoverflow.com/a/56691091
    https://blog.dataiku.com/narrowing-the-search-which-hyperparameters-really-matter (cites https://arxiv.org/abs/1710.04725)
"""

import argparse
import sys
import random
import warnings
import pandas                            
#import matplotlib.pyplot as plt    
import collections               
#from IPython.display import Image                 
#from IPython.display import display               
#from time import gmtime, strftime  
import numpy as np                
from numpy import mean
from numpy import std
from numpy import linspace
#from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
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
#from tqdm import tqdm
import random
xgb.set_config(verbosity=2)

'''
Function: load dataset. 

Args:
    filepath: A string consisting of the file path of the data set.
    predstartcol: An integer referring to the first column of the predictor variables. 
    predendcol: An integer referring to the last column of the predictor variables. Of course, this assumes that the target variable is not in the middle of the the predictor variables. ... 
    targetcol: An integer referring to the column of the data set containing the target variable.
    groupscol: An integer referring to the column of the data set containing the group memberships of each row (observation.)

Returns:
    X: An n x p matrix or vector of predictor variables.
    y: An n x 1 vector of the target variable. 
    groups: An n x 1 vector of the class membership of the entries in X, y. 
'''
def load_dataset(filepath, targetcol, groupscol, predstartcol, predendcol=-99):
    data = pandas.read_csv(filepath)
    
    if predendcol==-99:
        X = data.iloc[:,predstartcol:len(data.columns)]
    else:
        X = data.iloc[:,predstartcol:predendcol]
    y = data.iloc[:,targetcol]
    groups = data.iloc[:,groupscol]
    return X, y, groups

'''
Function: create a dict of standard models to evaluate {name:object}.
'''
def define_models(hp_type, models=collections.OrderedDict()):
    # Remove previously defined models.
    models.clear()
    
    if hp_type == "All":
        # Set up the hyperparameter grid.
        # Maximum depth per tree.
        # Per https://github.com/dmlc/xgboost/issues/4286 we should limit the depth to x, where 2^x < number of data points in the training set. 
        # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/ uses max_depth in [1,9] on a data set with 61 000 data points, so we'll do the same. 
        max_depth = [int(x) for x in linspace(8, 12, num = 5)]
        # Learning rate.
        learning_rate = np.arange(0.1,0.3,0.01)
        # Number of trees in ensemble.
        #num_round = [int(x) for x in linspace(start = 100, stop = 500, num = 10)]
        # Fraction of columns to be randomly sampled for each tree.
        colsample_bytree = np.arange(0.47,1.0,0.01)
        #colsample_bylevel = np.round(np.arange(0.1,1.0,0.01))
        # Fraction of observations to be sampled for each tree.
        subsample = np.arange(0.2,1.0,0.01)
        # Pseudo-regularization parameter. Originally np.arange(0.0,40.0,0.005)
        #gamma = np.arange(0.0,40.0,0.005)
        # Number (percentages) of samples required to form a leaf node. Not used b/c really no effect for our data.
        #min_child_weight = np.arange(0.0001, 0.5, 0.001)
    
       	# Create the random grid.
        random_grid = []    
        for i in max_depth :
            for j in learning_rate :
                for k in colsample_bytree : 
                    for l in subsample :
                        random_grid.append( ( ( (i, j, k, l) ) ) )
                                        
                            
              
        # print("Available combinations : ",  random_grid)

        # Generate all possible models using the parameter grid. 
        for r in range( len(random_grid) ) :        
            models['XGB-' + str(random_grid[r][0]) + '-' + str(random_grid[r][1]) + '-' + str(random_grid[r][2]) + '-' + str(random_grid[r][3])] = xgb.XGBRegressor(max_depth = random_grid[r][0], learning_rate = random_grid[r][1],  colsample_bytree = random_grid[r][2], subsample = random_grid[r][3])
                                                                                                                                                                                                                                            
        print('Defined %d models' % len(models))
        print()
    
    elif hp_type == "max_depth":
        # Hyperparameter "tuning" using the subsets (not full data set) to get good idea for range of parameters to use for full.
        # Maximum depth per tree.
        # Per https://github.com/dmlc/xgboost/issues/4286 we should limit the depth to x, where 2^x < number of data points in the training set. 
        # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/ uses max_depth in [1,9] on a data set with 61 000 data points, so we'll do the same. 
        max_depth = [int(x) for x in linspace(1, 17, num = 17)]
    
        # Create the parameter grid.
        parameter_grid = []    
        for i in max_depth :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-max_depth-' + str(parameter_grid[r])] = xgb.XGBRegressor(max_depth = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
    
    elif hp_type == "learning_rate":
        learning_rate = np.arange(0.0005,0.3,0.0005)
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in learning_rate :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-learning_rate-' + str(parameter_grid[r])] = xgb.XGBRegressor(learning_rate = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "num_round": 
        # Number of trees in ensemble.
        num_round = [int(x) for x in linspace(start = 100, stop = 500, num = 10)]
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in num_round :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-num_round-' + str(parameter_grid[r])] = xgb.XGBRegressor(num_round = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "colsample_bytree":
        # Fraction of columns to be randomly sampled for each tree.
        colsample_bytree = np.arange(0.1,1.0,0.01)
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in colsample_bytree :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-colsample_bytree-' + str(parameter_grid[r])] = xgb.XGBRegressor(colsample_bytree = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "colsample_bylevel": 
        colsample_bylevel = np.round(np.arange(0.1,1.0,0.01))
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in colsample_bylevel :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-colsample_bylevel-' + str(parameter_grid[r])] = xgb.XGBRegressor(colsample_bylevel = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "subsample":
        # Fraction of observations to be sampled for each tree.
        subsample = np.arange(0.01,1.0,0.01)
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in subsample :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-subsample-' + str(parameter_grid[r])] = xgb.XGBRegressor(subsample = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "gamma":
        # Pseudo-regularization parameter. Originally np.arange(0.0,40.0,0.005)
        gamma = np.arange(0.0,40.0,0.005)
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in gamma :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-gamma-' + str(parameter_grid[r])] = xgb.XGBRegressor(gamma = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    elif hp_type == "min_child_weight": 
        # Number (percentages) of samples required to form a leaf node.
        min_child_weight = np.arange(0.0001, 0.5, 0.001)
        
        # Create the parameter grid.
        parameter_grid = []    
        for i in min_child_weight :
            parameter_grid.append(i)
              
        # print("Available combinations : ",  random_grid)
        
        # Generate all possible models using the parameter grid. 
        for r in range( len(parameter_grid) ) :        
            models['XGB-min_child_weight-' + str(parameter_grid[r])] = xgb.XGBRegressor(min_child_weight = parameter_grid[r])
            
        print('Defined %d models' % len(models))
        print()
        
    return models

    '''
    # Set up the hyperparameter grid.
    # Number of trees in random forest.
    n_estimators = [int(x) for x in linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split.
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree.
    max_depth = [int(x) for x in linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node.
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node.
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree.
    bootstrap = [True, False]
    
   	# Create the random grid.
    random_grid = []    
    for i in n_estimators :        
        for j in max_features :
            for k in max_depth : 
                for l in min_samples_split :
                    for m in min_samples_leaf :
                        for n in bootstrap :
                            random_grid.append( ( ( ( ( ( i, j, k, l, m, n ) ) ) ) ) )
              
    # print("Available combinations : ",  random_grid)

    # Generate all possible models using the parameter grid. 
    for o in range( len(random_grid) ) :        
        models['RandomForest-' + str(random_grid[o][0]) + '-' + str(random_grid[o][1]) + '-' + str(random_grid[o][2]) + '-' + str(random_grid[o][3]) + '-' + str(random_grid[o][4]) + '-' + str(random_grid[o][5])] = RandomForestRegressor(n_estimators = random_grid[o][0], 
                                      max_features = random_grid[o][1],
                                      max_depth = random_grid[o][2],
                                      min_samples_split = random_grid[o][3],
                                      min_samples_leaf = random_grid[o][4],
                                      bootstrap = random_grid[o][5]
                                      )
    print('Defined %d models' % len(models))
    print()
    
    return models

    '''

'''
Function: create a dict of standard models to evaluate {name:object}.
'''
def define_models_max_depth(models=collections.OrderedDict()):
    # Set up the hyperparameter grid.
    # Maximum depth per tree.
    # Per https://github.com/dmlc/xgboost/issues/4286 we should limit the depth to x, where 2^x < number of data points in the training set. 
    # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/ uses max_depth in [1,9] on a data set with 61 000 data points, so we'll do the same. 
    max_depth = [int(x) for x in linspace(1, 9, num = 9)]
    
   	# Create the random grid.
    random_grid = []    
    for i in max_depth :
        random_grid.append(i)
              
    # print("Available combinations : ",  random_grid)

    # Generate all possible models using the parameter grid. 
    for r in range( len(random_grid) ) :        
        models['XGB-' + str(random_grid[r])] = xgb.XGBRegressor(max_depth = random_grid[r])
                                                                                                                                                                                                                                            
    print('Defined %d models' % len(models))
    print()
    
    return models
    
'''
Function: create a dict of standard models to evaluate {name:object}.
'''
def define_models_learning_rate(models=collections.OrderedDict()):
    # Set up the hyperparameter grid.
    # Maximum depth per tree.
    # Per https://github.com/dmlc/xgboost/issues/4286 we should limit the depth to x, where 2^x < number of data points in the training set. 
    # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/ uses max_depth in [1,9] on a data set with 61 000 data points, so we'll do the same. 
    learning_rate = np.arange(0.0005,0.3,0.0005)
    
   	# Create the random grid.
    random_grid = []    
    for i in learning_rate :
        random_grid.append(i)
              
    # print("Available combinations : ",  random_grid)

    # Generate all possible models using the parameter grid. 
    for r in range( len(random_grid) ) :        
        models['XGB-' + str(random_grid[r])] = xgb.XGBRegressor(learning_rate = random_grid[r])
                                                                                                                                                                                                                                            
    print('Defined %d models' % len(models))
    print()
    
    return models
    

'''
This function takes a data set, a vector of classes/groups (encoded as integers) in which each entry corresponds to a row in the data set (i.e. is the class for that row), and a number of folds into which to split the data set.

Args:  
    groups: This is a vector whose length is equal to the number of observations (rows) in the data set dataset. 
    folds: This is an integer telling how many folds to split the dataset _by groups_ into. For example, if length(unique(groups)), i.e. the number of groups, is 60 (to use R syntax), and folds is 10, then each fold will contain 6 groups. The training-and-testing process will be conducted 10 times, and each time, a different fold will be used as the test set.  

Returns:
    dataset_split: This is a list of the resulting folds, in which each fold consists of the row numbers (in the original data set) to be held out as a test set for that fold. 
    
'''
def cv_split_by_groups(groups, folds=10):
    dataset_split = list()
    
    unique_groups = list(set(groups)) # All the (unique) groups. The parameter groups is not unique. 
    n_groups = len(unique_groups) # The total number of groups at the start. 
    fold_size = int(n_groups / folds) # Size of each fold, in number of groups. 
    
    for i in range(folds): # For() loop going over each fold. 
        fold = list() # List containing the indices of the rows to be used in fold i.
        
        # Randomly select m groups for this fold, where m = fold_size (the number of groups in each fold.)
        random.seed(i)
        m_groups = random.sample(unique_groups, fold_size) # The unique groups themselves, not their indices!
        
        # Populate fold i with the indices of all rows in the data set belonging to the groups in m_groups.
        for j in m_groups:
            # Get all the rows belonging to group j.
            indices = [i for i, x in enumerate(groups) if x == j]
            fold.extend(indices)
            
            # Remove group j from the list of unique groups so they won't be used for subsequent folds. 
            unique_groups.remove(j)
            
        dataset_split.append(fold) # Append() here because we want dataset_split to be a list of lists. Extend() above (fold.extend()) because we want fold to be a simple vector (list) of indices without any kind of separation
        
    return dataset_split

'''
This function evaluates a list of models by calling robust_evaluate_model() on each model, which in turn calls evaluate_model(), which in turn calls score_model(). Whew! 

Args:
    X: An n x p matrix or vector of predictor variables.
    y: An n x 1 vector of the target variable.
    models: A list of models to evaluate.
    groups: An n x 1 vector consisting of the group memberships of each observation in X, y. 
    folds: The number of folds to divide X and y into. 
    scoring_function: A function to be used to score the performance of model model on data X, y. 
    greater_is_better: A Boolean variable indicating whether higher values returned by scoring_function indicate better model performance. 
    
Returns:
    results: A dict of {model:mean score}.
'''
def evaluate_models(X, y, models, groups, folds, scoring_function, max_evals = 60):
    results = dict()
    results_raw = pandas.DataFrame(index = list(models.keys()), columns = list(range(1, folds+1)))
    results_raw_by_group = pandas.DataFrame(index = list(models.keys()), columns = list(set(groups)))
        
    items = list(models.items())
    
    if(len(items) < max_evals):
        max_evals_final = len(items)
    else:
        max_evals_final = max_evals
        
    for i in range(max_evals_final): # len(items) instead of max_evals for grid search.
        if(max_evals_final < max_evals):
            # Use all the models.
            name, model = items[i]
        else:
            # Choose random hyperparameters.
            random.seed(i)
            name, model = random.sample(items, 1)[0] # items[i] instead of random.sample(items, 1)[0] for grid search.
        
        print('Hyperparameter tuning: evaluating model ' + str(i) + ' of ' + str(max_evals_final))
        
        # Evaluate the model.
        scores, group_scores = robust_evaluate_model(X, y, model, groups, folds, scoring_function, name)
        # Show process.
        if scores is not None:
            # Store summarized results in results.
            results[name] = scores
            mean_score, std_score = mean(scores), std(scores)
    
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
            print()
            
            # Store raw results in results_raw (average score over all groups for a given fold) and results_raw_by_group (individual score of each group.)
            results_raw.loc[name] = scores
            # Making the results_raw_by_group is going to be a little more involved. We have to make sure that that particular cell line was actually in a test set for that model (because of the way cv_split_by_groups works, not all cell lines will end up in a test set.)
            for group in results_raw_by_group.columns:
                # Check if that group is actually in group_scores.
                if group in group_scores:
                    results_raw_by_group.loc[name, group] = group_scores[group]
                else:
                    results_raw_by_group.loc[name, group] = 'NA'
            
        else:
            print('>%s: error' % name)
            print()
            
    return results, results_raw, results_raw_by_group

'''
This function robustly evaluates models. Traps exceptions and ignores warnings. Calls evaluate_model(). Args are all passed to evaluate_model() and are not used explicitly by robust_evaluate_model() itself.

Args:
    X: An n x p matrix or vector consisting of the predictor variables.
    y: An n x q matrix or vector consisting of the response (target) variables.
    model: A machine-learning model, such as those found in the scikit-learn package.
    groups: An n x 1 vector consisting of the group memberships of each observation in X (and y.)
    folds: The number of folds to divide X and y into. 
    scoring_function: A function to be used to score the performance of model model on data X, y. 
    greater_is_better: A Boolean variable indicating whether higher values returned by scoring_function indicate better model performance. 
    
Returns:
    scores: 
'''
def robust_evaluate_model(X, y, model, groups, folds, scoring_function, name): # For robust evaluation of models; traps exceptions and ignores warnings.
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores, group_scores = evaluate_model(X, y, model, groups, folds, scoring_function, name)
    except:
        scores = None
    return scores, group_scores

'''
This function trains one model and evaluates it via score_model(). For hyperparameter tuning: this is our objective function ("a function that takes in hyperparameters and returns a score we are trying to minimize or maximize"; see https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search#Four-parts-of-Hyperparameter-tuning.)


Args:
    X: An n x p matrix or vector consisting of the predictor variables.
    y: An n x q matrix or vector consisting of the response (target) variables.
    model: A machine-learning model, such as those found in the scikit-learn package.
    groups: An n x 1 vector consisting of the group memberships of each observation in X (and y.)
    folds: The number of folds to divide X and y into. 
    scoring_function: A function to be used to score the performance of model model on data X, y. 
    greater_is_better: A Boolean variable indicating whether higher values returned by scoring_function indicate better model performance. 
    
Returns:
    scores: List of scores of the estimator (determined by scoring_function) for each run of the cross validation (i.e. for each fold.)
'''
def evaluate_model(X, y, model, groups, folds, scoring_function, name):
    # Split training/test. 
    split = cv_split_by_groups(groups, folds)
    
    # Loop through each fold and build the model, make predictions, and then score the model. 
    fold_scores = list()
    all_indices = set(range(len(X)))
    group_scores = {}
    for i in range(len(split)): 
        test_indices = split[i]
        train_indices = list(all_indices.difference(test_indices))
        X_train = X.iloc[train_indices,:]
        y_train = y.iloc[train_indices] # y.iloc[train_indices,:]
        X_test = X.iloc[test_indices,:]
        y_test = y.iloc[test_indices] # y.iloc[test_indices,:]
        
        # Build model. 
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        print("Fitting model for fold %d of model %s" % (i, name))
        print("Time: " + current_time)

        model.fit(X_train, y_train) # Do only scikit-learn models have the .fit() method?
        
        # Use model to predict. 
        predicted = pandas.Series(model.predict(X_test))
        
        # Evaluate model.
        predicted = predicted.to_frame()
        y_test = y_test.to_frame()
        temp = y_test.join(predicted.set_index(y_test.index))
        group_scores_i = score_model(temp.iloc[:,1], temp.iloc[:,0], groups[test_indices], scoring_function)
        
        # Add mean_score (mean of scores of the individual groups in fold i) to fold_scores.
        fold_scores.append(stats.mean(group_scores_i.values()))
        # Add individual group scores to group_scores.
        group_scores.update(group_scores_i)
    
    return fold_scores, group_scores
        
'''
This is the function called by evaluate_model() to score the performance of one model, evaluating each group in the test set individually. 

Args:
    y_true: An n x 1 vector. The ground-truth values.
    y_pred: An n x 1 vector. The values predicted by the model.
    groups: An n x 1 vector consisting of the group memberships of each observation in X (and y.)
    scoring_function: A function to be used to score the performance of model model on data X, y. 
    greater_is_better: A Boolean variable indicating whether higher values returned by scoring_function indicate better model performance.
    
Returns:
    mean_score: A numeric value. The mean score over all the groups in a given fold. 
    SD_score: A numeric value. The standard deviation of the scores of all the groups in a given fold. 
'''    
def score_model(y_true, y_pred, groups_sub, scoring_function):    
    group_scores = {}
    # For each group, score the performance of the model using the scoring function defined by scoring_function. 
    unique_groups = list(set(groups_sub))
    for i in unique_groups:
        # Get the indices of y_true, y_pred that correspond to group i. 
        indices =  groups_sub[groups_sub==i].index # [j for j, x in enumerate(groups_sub) if x == i] <- This doesn't work for Pandas series, because it resets the indices. 
        # Subset y_true, y_pred to include only entries in group i. 
        y_true_i = y_true[indices] # Can we do it this way? Yes, if this is a series object of pandas.core.series module. If this is a dataframe, we will need to use .iloc[:,:], of course.
        y_pred_i = y_pred[indices]
        
        # Score and add the score to the scores list.
        group_scores[i] = scoring_function(y_true_i, y_pred_i) # When group_scores was a list: group_scores.append(scoring_function(y_true_i, y_pred_i))
    
    # mean_score = stats.mean(group_scores.values())
    # SD_score = stats.stdev(group_scores)
    
    return group_scores # mean_score is the average score across the groups.

''' 
This is our custom scoring function. It uses Spearman rank correlation. 

Args:
    y_true: An n x 1 vector. The ground-truth vector; the actual experimentally determined values.
    y_pred: An n x 1 vector. The values predicted by the model. 
    
Returns:
    cor: A numeric value. The Spearman rank correlation of y_true and y_pred. The higher the value, the better the performance of the model being evaluated. 
'''
def correlation_coefficient_scoring_function(y_true, y_pred):
    cor, p = spearmanr(y_true, y_pred)
    return(cor)

'''
This is our function to summarize results.

Args:
    results: A dict of {model:mean score}, generated by evaluate_models().
    maximize: A Boolean variable telling the function whether higher scores are better.
    top_n: The number of best-performing models to include. 
    fig_path: The directory in which to save the final results. 
    
Returns:
    [Does not return a value.]
'''
def summarize_results(results, fig_name, maximize = True, top_n = 10):
    # Check for no results.
    if len(results)==0:
        print()
        print('No results.')
        return
    # Determine how many results to summarize. 
    n  = min(top_n, len(results))
    # Create a list of (name, mean(scores)) tuples. 
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # Sort tuples by mean score.
    mean_scores = sorted(mean_scores, key = lambda x: x[1])
    # Reverse for descending order (e.g. for accuracy.)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # Retrieve the top n for summarization.
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # Print the top n.
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
    # Boxplot for the top n.
    #pyplot.boxplot(scores, labels = names)
    #_, labels = pyplot.xticks()
    #pyplot.setp(labels, rotation=90)
    #pyplot.savefig(fig_name)
    
'''
This is our main function.

Args:
    argv
        dataset_filename: The path to the CSV file containing the training data.
        target_column: The column in data that contains the target variable. 
        raw_results_filename: The full filename to which the raw results should be saved.
        raw_results_by_group_filename: The full filename to which the raw results by group should be saved. 
        results_summary_fig_filename: The full filename to which a summary figure should be saved. 
    
Returns: 
    [Does not return a value.]
'''

def main():
    # https://stackoverflow.com/a/24180537

    # Get the arguments.
    dataset_filename = ''
    target_column = ''
    pred_start_column = ''
    grouping_column = ''
    raw_results_filename = ''
    raw_results_by_group_filename = ''
    num_folds = ''
    ht_type = ''
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dataset',
                    required=True,
                    type=str, 
                    dest="dataset",
                    metavar="<path to data set>",
                    help="Full path to the data set" )
        parser.add_argument('-t', '--targetcol',
                    required=True,
                    type=int, 
                    dest="targetcol",
                    metavar="<target column>",
                    help="Number corresponding to the column containing the target variable" )
        parser.add_argument('-s', '--predstartcol',
                    required=True,
                    type=int, 
                    dest="predstartcol",
                    metavar="<predictor start column>",
                    help="Number indicating the first column containing predictor variables" )
        parser.add_argument('-e', '--predendcol',
                    required=False,
                    type=int, 
                    default=-99,
                    dest="predendcol",
                    metavar="<predictor end column>",
                    help="Number indicating the last column containing predictor variables" )
        parser.add_argument('-g', '--groupingcol',
                    required=True,
                    type=int, 
                    dest="groupingcol",
                    metavar="<grouping column>",
                    help="Number corresponding to the column containing the grouping variable" )
        parser.add_argument('-r', '--rawresults',
                    required=True,
                    type=str, 
                    dest="rawresults",
                    metavar="<path to raw results file>",
                    help="Full path to the raw results file to be generated" )
        parser.add_argument('-b', '--rawresultsbygroup',
                    required=True,
                    type=str, 
                    dest="rawresultsbygroup",
                    metavar="<path to raw results by group file>",
                    help="Full path to the raw results (by group) file to be generated" )
        #parser.add_argument('-f', '--summaryfig',
        #            required=True,
        #            type=str, 
        #            dest="summaryfig",
        #            metavar="<path to summary figure file>",
        #            help="Full path to the summary figure file to be generated" )
        parser.add_argument('-n', '--numfolds',
                    required=False,
                    type=int, 
                    dest="numfolds",
                    default=3,
                    metavar="<number of folds>",
                    help="Number of folds during training" )
        parser.add_argument('-p', '--httype',
                    required=False,
                    type=str, 
                    dest="httype",
                    default="Spotcheck",
                    metavar="<full or spotcheck tuning>",
                    help="Whether to do hyperparameter tuning for the full model (data set) or for the spotchecking data sets" )
        args = parser.parse_args()
        
    except argparse.ArgumentError:
        print('model_spot-checking.py -d <dataset> -t <targetcol> -s <predstartcol> -g <groupingcol> -r <rawresults> -b <rawresultsbygroup> -f <summaryfig> [-n <numfolds>] [-e <predendcol>]')
        sys.exit(2)
    
    dataset_filename = args.dataset
    target_column = args.targetcol
    pred_start_column = args.predstartcol
    pred_end_column = args.predendcol
    grouping_column = args.groupingcol
    raw_results_filename = args.rawresults
    raw_results_by_group_filename = args.rawresultsbygroup
    #results_summary_fig_filename = args.summaryfig
    num_folds = args.numfolds
    ht_type = args.httype

    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Hyperparameter-tuning for dataset file " + dataset_filename + " beginning at " + current_time)

    # Load dataset.
    X, y, groups = load_dataset(dataset_filename, target_column, grouping_column, pred_start_column, pred_end_column) # Remember that with Python, numbering begins at 0.

    ''' 
    As of 2021-11-06, we are not scaling predictors. I think it might affect individual predictions using the trained model. ...
    # Scale predictor values.
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    # Why we don't need to scale target values: https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X_scaled = scaling.transform(X)
    X_scaled = pandas.DataFrame(X_scaled) # This part necessary because scaling transforms it into a numpy ndarray, and later functions require a Pandas DataFrame as input. 
    '''
    
    # Get model list.
    if ht_type == "Spotcheck":
        hyperparameters = ["max_depth", "learning_rate", "subsample", "min_child_weight", "colsample_bytree"]
        for hyperparameter in hyperparameters:
            models = define_models(hp_type = hyperparameter)
            
            # Evaluate models.
            results, results_raw, results_raw_by_group = evaluate_models(X = X, y = y, models = models, groups = groups, folds = num_folds, scoring_function = correlation_coefficient_scoring_function) # num_folds default is 3. 
            # Save raw results.
            raw_results_filename_final = raw_results_filename + "_" + hyperparameter + ".csv"
            raw_results_by_group_filename_final = raw_results_by_group_filename + "_" + hyperparameter + ".csv"
            results_raw.to_csv(raw_results_filename_final)
            results_raw_by_group.to_csv(raw_results_by_group_filename_final)
            
    else:
        models = define_models(hp_type = "All")
            
        # Evaluate models.
        results, results_raw, results_raw_by_group = evaluate_models(X = X, y = y, models = models, groups = groups, folds = num_folds, scoring_function = correlation_coefficient_scoring_function) # num_folds default is 3. 
        # Save raw results.
        raw_results_filename_final = raw_results_filename + ".csv"
        raw_results_by_group_filename_final = raw_results_by_group_filename + ".csv"
        results_raw.to_csv(raw_results_filename_final)
        results_raw_by_group.to_csv(raw_results_by_group_filename_final)

    # Mark time at finish of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Hyperparameter-tuning concluding at " + current_time)

if __name__ == "__main__":
    main()
