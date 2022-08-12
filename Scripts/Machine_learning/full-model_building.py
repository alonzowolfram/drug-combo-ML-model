# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:21:17 2020

@author: lwfong
"""

import argparse
import sys
import random
import warnings
import pandas                            
#import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                  
from numpy import mean
from numpy import std
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
import pickle
xgb.set_config(verbosity=2)

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
    metadata = data.iloc[:,0:predstartcol] # Range is not inclusive at the upper end: [).
    return X, y, groups, metadata

'''
Create a dict of standard models to evaluate {name:object}.
'''
def define_models(models=dict()):
	
	#n_trees = 100 #100
	#models['AdaBoost'] = AdaBoostRegressor(n_estimators=n_trees)
	#models['Bagging'] = BaggingRegressor(n_estimators=n_trees, verbose = 11) # Verbose
	#models['RandomForest'] = RandomForestRegressor(n_estimators=n_trees, verbose = 11) # Verbose
	#models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=n_trees, verbose = 11) # Verbose
	#models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=n_trees, verbose = 11) # Verbose
	models['XGB'] = xgb.XGBRegressor() # Verbose.
    
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
This function takes a data set, a vector of classes/groups (encoded as integers) in which each entry corresponds to a row in the data set (i.e. is the class for that row), and a number of folds into which to split the data set.

Args:  
    groups: This is a vector whose length is equal to the number of observations (rows) in the data set dataset. 
    test_train_ratio: This is a float indicating the ratio of the test set to the training set (usually 1:5). 

Returns:
    dataset_split: This is a list of the resulting test fold (length 1), in which each fold (there's only 1) consists of the row numbers (in the original data set) to be held out as a test set for that fold. Here the list will only have one element, a list with test_size elements. 
    
'''
def train_test_split_by_groups(groups, test_train_ratio=0.2):
    dataset_split = list()
    
    unique_groups = list(set(groups)) # All the (unique) groups. The parameter groups is not unique. 
    n_groups = len(unique_groups) # The total number of groups at the start. 
    test_size = int(n_groups * test_train_ratio) # Size of the training set, in number of groups. 
    
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
            
    dataset_split.append(test_set) # Append() here because we want dataset_split to be a list of lists. Extend() above (fold.extend()) because we want fold to be a simple vector (list) of indices without any kind of separation
        
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
def evaluate_models(X, y, models, groups, folds, scoring_function, metadata):
    results = dict()
    results_raw = pandas.DataFrame(index = list(models.keys()), columns = list(range(1, folds+1)))
    results_raw_by_group = pandas.DataFrame(index = list(models.keys()), columns = list(set(groups)))
    predictions = []
    
    items = list(models.items())
    for i in range(0,1): 
        name, model = items[i]
        
        print(name)
        print('Full-dataset cross-validation: evaluating model ' + str(i+1) + ' of ' + str(len(range(0,1))))
        
        # Evaluate the model.
        scores, group_scores, model_predictions = robust_evaluate_model(X, y, model, groups, folds, scoring_function, name, metadata)
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
                    
            # Store predictions in predictions variable.
            # Create a Series consisted of the model name repeated n times.
            model_name = pandas.Series(name)
            model_series = model_name.repeat(len(model_predictions))
            # Reindex or we'll get a "Reindexing only valid with uniquely valued Index objects" error.
            # https://stackoverflow.com/a/45056184
            model_series = model_series.reset_index(drop=True)
            model_predictions = model_predictions.reset_index(drop=True)
            # Put it all together.
            model_predictions_labeled = pandas.concat([model_series, model_predictions], axis = 1) # Labeled with the model.
            predictions.append(model_predictions_labeled)
            
        else:
            print('>%s: error' % name)
            print()
            
    predictions_final = pandas.concat(predictions)
    return results, results_raw, results_raw_by_group, predictions_final

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
def robust_evaluate_model(X, y, model, groups, folds, scoring_function, name, metadata): # For robust evaluation of models; traps exceptions and ignores warnings.
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores, group_scores, model_predictions = evaluate_model(X, y, model, groups, folds, scoring_function, name, metadata)
    except:
        scores = None
    return scores, group_scores, model_predictions

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
def evaluate_model(X, y, model, groups, folds, scoring_function, name, metadata):
    # Split training/test. 
    split = cv_split_by_groups(groups, folds)
    
    # Loop through each fold and build the model, make predictions, and then score the model. 
    fold_scores = list()
    all_indices = set(X.index) #set(range(len(X)))
    group_scores = {}
    predictions = []
    for i in range(len(split)): 
        test_indices = split[i]
        train_indices = list(all_indices.difference(test_indices))
        X_train = X.loc[train_indices,:]
        y_train = y.loc[train_indices] # y.iloc[train_indices,:]
        X_test = X.loc[test_indices,:]
        y_test = y.loc[test_indices] # y.iloc[test_indices,:]
        metadata_test = metadata.loc[test_indices,:]
        # Why we're using loc and not iloc: https://stackoverflow.com/questions/44123056/indexerror-positional-indexers-are-out-of-bounds-when-theyre-demonstrably-no/44123390
        
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
        
        # Create a data frame consisting of the scores and the metadata (the cell lines and drug combinations.)
        # But re-index first, because metadata_test and predicted have different indices!!
        # https://stackoverflow.com/a/45056184
        metadata_test = metadata_test.reset_index(drop=True)
        predicted = predicted.reset_index(drop=True)
        predictions_i = pandas.concat([metadata_test, predicted], axis=1)
        # Append to predictions list.
        predictions.append(predictions_i)
    
    # Concatenate all the data frames in predictions.
    model_predictions = pandas.concat(predictions)
    
    return fold_scores, group_scores, model_predictions
        
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
        print(i)
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
    test_scores_filename = ''
    test_scores_by_group_filename = ''
    final_model_filename = ''
    num_folds = ''
    
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
                    help="Full path to the raw cross-validation results file to be generated" )
        parser.add_argument('-b', '--rawresultsbygroup',
                    required=True,
                    type=str, 
                    dest="rawresultsbygroup",
                    metavar="<path to raw results by group file>",
                    help="Full path to the raw cross-validation results (by group) file to be generated" )
        parser.add_argument('-k', '--predictions',
                    required=True,
                    type=str, 
                    dest="predictions",
                    metavar="<path to predictions file>",
                    help="Full path to the predictions file to be generated" )
        parser.add_argument('-n', '--numfolds',
                    required=False,
                    type=int, 
                    dest="numfolds",
                    default=3,
                    metavar="<number of folds>",
                    help="Number of folds during training" )
        parser.add_argument('-c', '--testscores',
                    required=True,
                    type=str, 
                    dest="testscores",
                    metavar="<test scores filename>",
                    help="Full path to the file containing the by-fold scores for the test results" )
        parser.add_argument('-q', '--grouptestscores',
                    required=True,
                    type=str, 
                    dest="grouptestscores",
                    metavar="<group test scores filename>",
                    help="Full path to the file containing the by-group scores for the test results" )
        parser.add_argument('-f', '--finalmodel',
                    required=True,
                    type=str, 
                    dest="finalmodel",
                    metavar="<final model filename>",
                    help="Full path to the final model pickle file" )
        args = parser.parse_args()
        
    except argparse.ArgumentError:
        print('full-model_building.py -d <dataset> -t <targetcol> -s <predstartcol> -g <groupingcol> -r <rawresults> -b <rawresultsbygroup> -f <summaryfig> [-n <numfolds>] [-e <predendcol>]')
        sys.exit(2)
    
    dataset_filename = args.dataset
    target_column = args.targetcol
    pred_start_column = args.predstartcol
    pred_end_column = args.predendcol
    grouping_column = args.groupingcol
    raw_results_filename = args.rawresults
    raw_results_by_group_filename = args.rawresultsbygroup
    predictions_final_filename = args.predictions
    test_scores_filename = args.testscores
    test_scores_by_group_filename = args.grouptestscores
    final_model_filename = args.finalmodel
    num_folds = args.numfolds

    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Model-building for dataset file " + dataset_filename + " beginning at " + current_time)

    # Load dataset.
    X, y, groups, metadata = load_dataset(dataset_filename, target_column, grouping_column, pred_start_column, pred_end_column) # Remember that with Python, numbering begins at 0.
    
    
    ''' 
    As of 2021-11-06, we are not scaling predictors. I think it might affect individual predictions using the trained model. ...
    # Scale predictor values.
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    # Why we don't need to scale target values: https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X_scaled = scaling.transform(X)
    X_scaled = pandas.DataFrame(X_scaled) # This part necessary because scaling transforms it into a numpy ndarray, and later functions require a Pandas DataFrame as input. 
    '''
    
    # Split into training and test. 
    test_train_split = train_test_split_by_groups(groups, 0.2)
    all_indices = set(range(len(X))) 
    
    test_indices = test_train_split[0]
    train_indices = list(all_indices.difference(test_indices))
    X_scaled_train = X.iloc[train_indices,:]
    y_train = y.iloc[train_indices] # y.iloc[train_indices,:]
    X_scaled_test = X.iloc[test_indices,:]
    y_test = y.iloc[test_indices] # y.iloc[test_indices,:]
    #groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    #metadata_train = metadata.iloc[train_indices,:]
    #metadata_test = metadata.iloc[test_indices,:]
    
    # Get model list.
    models = define_models()

    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Model evaluation beginning at " + current_time)

    # Evaluate the model on the training set.
    # Perform cross-validation.
    results, results_raw, results_raw_by_group, predictions_final = evaluate_models(X = X, y = y, models = models, groups = groups, folds = num_folds, scoring_function = correlation_coefficient_scoring_function, metadata = metadata) # num_folds default is 3. 
    # Save raw results (CV).
    results_raw.to_csv(raw_results_filename)
    results_raw_by_group.to_csv(raw_results_by_group_filename)
    predictions_final.to_csv(predictions_final_filename)
    
    # Test on the test set (X_scaled_test).
    test_scores = list()
    group_test_scores = {}
    #model_type = 'XGBoost'
    model = xgb.XGBRegressor()
    model.fit(X_scaled_train, y_train) # Do only scikit-learn models have the .fit() method?
    # Use model to predict. 
    predicted = pandas.Series(model.predict(X_scaled_test))
    # Evaluate model.
    predicted = predicted.to_frame()
    y_test = y_test.to_frame()
    temp = y_test.join(predicted.set_index(y_test.index))
    group_test_scores_i = score_model(temp.iloc[:,1], temp.iloc[:,0], groups_test, scoring_function = correlation_coefficient_scoring_function)
    # Add mean_score (mean of scores of the individual groups in fold i) to fold_scores.
    test_scores.append(stats.mean(group_test_scores_i.values()))
    # Add individual group scores to group_scores.
    group_test_scores.update(group_test_scores_i)
    pandas.DataFrame(test_scores).to_csv(test_scores_filename)
    pandas.DataFrame(group_test_scores, index=[0]).to_csv(test_scores_by_group_filename)
    print("Model evaluation concluding at " + current_time)
    
    # Fit the final model on the complete set.
    print("Final model building beginning at " + current_time)
    model.fit(X, y)
    # Save the model to disk.
    pickle.dump(model, open(final_model_filename, 'wb'))
    
    # Mark time at end of model-building.
    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print("Final model building concluding at " + current_time)

if __name__ == "__main__":
    main()
    
'''
https://www.csestack.org/python-list-installed-modules-versions-pip/
https://stackoverflow.com/questions/18966564/pip-freeze-vs-pip-list
TACC Python and Python package versions as of 06/16/2020:
    
Python 3.7.4
Package                            Version  
---------------------------------- ---------
alabaster                          0.7.12   
anaconda-client                    1.7.2    
anaconda-navigator                 1.9.7    
anaconda-project                   0.8.3    
asn1crypto                         1.0.1    
astroid                            2.3.1    
astropy                            3.2.2    
atomicwrites                       1.3.0    
attrs                              19.2.0   
Babel                              2.7.0    
backcall                           0.1.0    
backports.functools-lru-cache      1.5      
backports.os                       0.1.1    
backports.shutil-get-terminal-size 1.0.0    
backports.tempfile                 1.0      
backports.weakref                  1.0.post1
beautifulsoup4                     4.8.0    
bitarray                           1.0.1    
bkcharts                           0.2      
bleach                             3.1.0    
bokeh                              1.3.4    
boto                               2.49.0   
Bottleneck                         1.2.1    
certifi                            2019.9.11
cffi                               1.12.3   
chardet                            3.0.4    
Click                              7.0      
cloudpickle                        1.2.2    
clyent                             1.2.2    
colorama                           0.4.1    
conda                              4.7.12   
conda-build                        3.18.9   
conda-package-handling             1.6.0    
conda-verify                       3.4.2    
contextlib2                        0.6.0    
cryptography                       2.7      
cycler                             0.10.0   
Cython                             0.29.13  
cytoolz                            0.10.0   
dask                               2.5.2    
decorator                          4.4.0    
defusedxml                         0.6.0    
distributed                        2.5.2    
docutils                           0.15.2   
entrypoints                        0.3      
et-xmlfile                         1.0.1    
fastcache                          1.1.0    
filelock                           3.0.12   
Flask                              1.1.1    
fsspec                             0.5.2    
future                             0.17.1   
gevent                             1.4.0    
glob2                              0.7      
gmpy2                              2.0.8    
greenlet                           0.4.15   
h5py                               2.8.0    
HeapDict                           1.0.1    
html5lib                           1.0.1    
idna                               2.8      
imageio                            2.6.0    
imagesize                          1.1.0    
importlib-metadata                 0.23     
ipykernel                          5.1.2    
ipython                            7.8.0    
ipython-genutils                   0.2.0    
ipywidgets                         7.5.1    
isort                              4.3.21   
itsdangerous                       1.1.0    
jdcal                              1.4.1    
jedi                               0.15.1   
jeepney                            0.4.1    
Jinja2                             2.10.3   
joblib                             0.13.2   
json5                              0.8.5    
jsonschema                         3.0.2    
jupyter                            1.0.0    
jupyter-client                     5.3.3    
jupyter-console                    6.0.0    
jupyter-core                       4.5.0    
jupyterlab                         1.1.4    
jupyterlab-server                  1.0.6    
keyring                            18.0.0   
kiwisolver                         1.1.0    
lazy-object-proxy                  1.4.2    
libarchive-c                       2.8      
lief                               0.9.0    
llvmlite                           0.29.0   
locket                             0.2.0    
lxml                               4.4.1    
MarkupSafe                         1.1.1    
matplotlib                         3.1.1    
mccabe                             0.6.1    
mistune                            0.8.4    
mkl-fft                            1.0.14   
mkl-random                         1.1.0    
mkl-service                        2.3.0    
mock                               3.0.5    
more-itertools                     7.2.0    
mpi4py                             3.0.0    
mpmath                             1.1.0    
msgpack                            0.6.1    
multipledispatch                   0.6.0    
navigator-updater                  0.2.1    
nbconvert                          5.6.0    
nbformat                           4.4.0    
networkx                           2.3      
nltk                               3.4.5    
nose                               1.3.7    
notebook                           6.0.1    
numba                              0.45.1   
numexpr                            2.7.0    
numpy                              1.17.2   
numpydoc                           0.9.1    
olefile                            0.46     
openpyxl                           3.0.0    
packaging                          19.2     
pandas                             0.25.1   
pandocfilters                      1.4.2    
parso                              0.5.1    
partd                              1.0.0    
path.py                            12.0.1   
pathlib2                           2.3.5    
patsy                              0.5.1    
pep8                               1.7.1    
pexpect                            4.7.0    
pickleshare                        0.7.5    
Pillow                             6.2.0    
pip                                19.2.3   
pkginfo                            1.5.0.1  
pluggy                             0.13.0   
ply                                3.11     
prometheus-client                  0.7.1    
prompt-toolkit                     2.0.10   
psutil                             5.6.3    
ptyprocess                         0.6.0    
py                                 1.8.0    
pycodestyle                        2.5.0    
pycosat                            0.6.3    
pycparser                          2.19     
pycrypto                           2.6.1    
pycurl                             7.43.0.3 
pyflakes                           2.1.1    
Pygments                           2.4.2    
pylint                             2.4.2    
pyodbc                             4.0.27   
pyOpenSSL                          19.0.0   
pyparsing                          2.4.2    
pyrsistent                         0.15.4   
PySocks                            1.7.1    
pytest                             5.2.1    
pytest-arraydiff                   0.3      
pytest-astropy                     0.5.0    
pytest-doctestplus                 0.4.0    
pytest-openfiles                   0.4.0    
pytest-remotedata                  0.3.2    
python-dateutil                    2.8.0    
pytz                               2019.3   
PyWavelets                         1.0.3    
PyYAML                             5.1.2    
pyzmq                              18.1.0   
QtAwesome                          0.6.0    
qtconsole                          4.5.5    
QtPy                               1.9.0    
requests                           2.22.0   
rope                               0.14.0   
ruamel-yaml                        0.15.46  
scikit-image                       0.15.0   
scikit-learn                       0.21.3   
scipy                              1.3.1    
seaborn                            0.9.0    
SecretStorage                      3.1.1    
Send2Trash                         1.5.0    
setuptools                         41.4.0   
simplegeneric                      0.8.1    
singledispatch                     3.4.0.3  
six                                1.12.0   
sklearn                            0.0      
snowballstemmer                    2.0.0    
sortedcollections                  1.1.2    
sortedcontainers                   2.1.0    
soupsieve                          1.9.3    
Sphinx                             2.2.0    
sphinxcontrib-applehelp            1.0.1    
sphinxcontrib-devhelp              1.0.1    
sphinxcontrib-htmlhelp             1.0.2    
sphinxcontrib-jsmath               1.0.1    
sphinxcontrib-qthelp               1.0.2    
sphinxcontrib-serializinghtml      1.1.3    
sphinxcontrib-websupport           1.1.2    
spyder                             3.3.6    
spyder-kernels                     0.5.2    
SQLAlchemy                         1.3.9    
statsmodels                        0.10.1   
sympy                              1.4      
tables                             3.5.2    
tblib                              1.4.0    
terminado                          0.8.2    
testpath                           0.4.2    
toolz                              0.10.0   
tornado                            6.0.3    
tqdm                               4.36.1   
traitlets                          4.3.3    
unicodecsv                         0.14.1   
urllib3                            1.24.2   
wcwidth                            0.1.7    
webencodings                       0.5.1    
Werkzeug                           0.16.0   
wheel                              0.33.6   
widgetsnbextension                 3.5.1    
wrapt                              1.11.2   
wurlitzer                          1.0.3    
xgboost                            0.90     
xlrd                               1.2.0    
XlsxWriter                         1.2.1    
xlwt                               1.3.0    
zict                               1.0.0    
zipp                               0.6.0  
'''