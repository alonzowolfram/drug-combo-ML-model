#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:12:59 2020

@author: alonzowolf
"""

'''
Edit before running script: 
    1) Log filename
    2) Results filenames (PNG, overall, and by group)
    3) Whether to use BestComboScore or BestPercentGrowth as the target variable
    4) Model(s) to be used
    
'''

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

'''
Load dataset. 

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
Create a dict of standard models to evaluate {name:object}.
'''
def define_models(models=dict()):
    
   	# Linear models
	#models['lr'] = LinearRegression()
	#alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	#for a in alpha:
	#	models['lasso-'+str(a)] = Lasso(alpha=a)
    #alpha = [0.4, 0.9]
    #for a in alpha:
    #    models['ridge-'+str(a)] = Ridge(alpha=a)
	#for a1 in alpha:
	#	for a2 in alpha:
	#		name = 'en-' + str(a1) + '-' + str(a2)
	#		models[name] = ElasticNet(a1, a2)
    models['en-0.0-0.4'] = ElasticNet(0.0, 0.4)
	#models['huber'] = HuberRegressor()
	#models['lars'] = Lars()
	#models['llars'] = LassoLars()
	#models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	#models['ranscac'] = RANSACRegressor()
	#models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	#models['theil'] = TheilSenRegressor() # Doesn't work. 
	
    # Non-linear models
	#n_neighbors = range(1, 21)
	#for k in n_neighbors:
	#	models['knn-'+str(k)] = KNeighborsRegressor(n_neighbors=k)
	#models['cart'] = DecisionTreeRegressor()
	#models['extra'] = ExtraTreeRegressor()
	#models['svml'] = SVR(kernel='linear') # Doesn't work.
	#models['svmp'] = SVR(kernel='poly') # Doesn't work. 
	#c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	#for c in c_values:
	#	models['svmr'+str(c)] = SVR(C=c)
	
    # Ensemble models
    #n_trees = 100
	#models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
    #models['bag'] = BaggingRegressor(n_estimators=n_trees)
	#models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	#models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	#models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	#models['xgb'] = xgb.XGBRegressor()
    
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
def evaluate_models(X, y, models, groups, folds, scoring_function):
    results = dict()
    results_raw = pandas.DataFrame(index = list(models.keys()), columns = list(range(1, folds+1)))
    results_raw_by_group = pandas.DataFrame(index = list(models.keys()), columns = list(set(groups)))
    
    for name, model in models.items():
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
This function trains one model and evaluates it via score_model().

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
    pyplot.boxplot(scores, labels = names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    pyplot.savefig(fig_name)

# Establish connection to the log file. 
today = date.today()
date = today.strftime("%b-%d-%Y")

# Mark time at beginning of model-building. 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print()
print("Model-building beginning at " + current_time)


# Load dataset.
X, y, groups = load_dataset('~/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Data/Training_data/complete_dataset_PaDEL_L1000_cs_pg.csv', 4, 1, 6, -99) # 4 = fifth column = PercentGrowth. 

# Get model list.
models = define_models()

# Evaluate models.
results, results_raw, results_raw_by_group = evaluate_models(X = X, y = y, models = models, groups = groups, folds = 10, scoring_function = correlation_coefficient_scoring_function)

# Summarize results.
results_summary_fig_filename = 'complete_en_PaDEL_test.png' # 'complete_dataset_comparison_PaDEL.png' 
summarize_results(results, results_summary_fig_filename)

# Save raw results.
raw_results_filename = '~/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Results/Cell_growth/Model_scores/complete_en_PaDEL_results_raw_test.csv' 
results_raw.to_csv(raw_results_filename)
raw_results_by_group_filename = '~/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Results/Cell_growth/Model_scores/complete_en_PaDEL_results_raw_by_group_test.csv' 
results_raw_by_group.to_csv(raw_results_by_group_filename)

# Mark time at finish of model-building. 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print()
print("Model-building concluding at " + current_time)