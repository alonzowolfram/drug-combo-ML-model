#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:12:59 2020

@author: alonzowolf
"""

import argparse
import sys
import random
import warnings
import pandas                                            
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import statistics as stats
from datetime import datetime
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
def load_dataset(filepath, targetcol, predstartcol, predendcol=-99):
    data = pandas.read_csv(filepath)
    
    if predendcol==-99:
        X = data.iloc[:,predstartcol:len(data.columns)]
    else:
        X = data.iloc[:,predstartcol:predendcol]
    y = data.iloc[:,targetcol]
    return X, y

'''
Create a dict of standard models to evaluate {name:object}.
'''
def define_models(models=dict()):
    
   	# Linear models
	models['Logistic'] = LogisticRegression()
	alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for a in alpha:
		models['Ridge-'+str(a)] = RidgeClassifier(alpha=a)
	models['SGD'] = SGDClassifier(max_iter=1000, tol=1e-3)
	models['PassAgg'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
	
    # Non-linear models
	n_neighbors = range(1, 21)
	for k in n_neighbors:
		models['KNN-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
	models['DecisionTree'] = DecisionTreeClassifier()
	models['ExtraTree'] = ExtraTreeClassifier()
	models['SVML'] = SVC(kernel='linear')
	models['SVMP'] = SVC(kernel='poly')
	c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for c in c_values:
		models['SVMR'+str(c)] = SVC(C=c)
	models['Bayes'] = GaussianNB()
	
    # Ensemble models
	n_trees = 100
	models['AdaBoost'] = AdaBoostClassifier(n_estimators=n_trees)
	models['Bagging'] = BaggingClassifier(n_estimators=n_trees)
	models['RandomForest'] = RandomForestClassifier(n_estimators=n_trees)
	models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=n_trees)
	models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=n_trees)
	models['XGBoost'] = xgb.XGBClassifier()
	print('Defined %d models' % len(models))
	return models
    
	print('Defined %d models' % len(models))
	print()
    
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	#steps.append(('standardize', StandardScaler()))
	# normalization
	#steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
	# create the pipeline
	pipeline = make_pipeline(model)
	# evaluate model
	scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
	return scores
 
# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, y, model, folds, metric)
	except:
		scores = None
	return scores
 
# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='accuracy'):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		scores = robust_evaluate_model(X, y, model, folds, metric)
		# show process
		if scores is not None:
			# store a result
			results[name] = scores
			mean_score, std_score = mean(scores), std(scores)
			print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
		else:
			print('>%s: error' % name)
	return results

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
    #grouping_column = ''
    raw_results_filename = ''
    #raw_results_by_group_filename = ''
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
        #parser.add_argument('-g', '--groupingcol',
        #            required=True,
        #            type=int, 
        #            dest="groupingcol",
        #            metavar="<grouping column>",
        #            help="Number corresponding to the column containing the grouping variable" )
        parser.add_argument('-r', '--rawresults',
                    required=True,
                    type=str, 
                    dest="rawresults",
                    metavar="<path to raw results file>",
                    help="Full path to the raw results file to be generated" )
        #parser.add_argument('-b', '--rawresultsbygroup',
        #            required=True,
        #            type=str, 
        #            dest="rawresultsbygroup",
        #            metavar="<path to raw results by group file>",
        #            help="Full path to the raw results (by group) file to be generated" )
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
        args = parser.parse_args()
        
    except argparse.ArgumentError:
        print('model_spot-checking.py -d <dataset> -t <targetcol> -s <predstartcol> -g <groupingcol> -r <rawresults> -b <rawresultsbygroup> -f <summaryfig> [-n <numfolds>] [-e <predendcol>]')
        sys.exit(2)
    
    dataset_filename = args.dataset
    target_column = args.targetcol
    pred_start_column = args.predstartcol
    pred_end_column = args.predendcol
    #grouping_column = args.groupingcol
    raw_results_filename = args.rawresults
    #raw_results_by_group_filename = args.rawresultsbygroup
    #results_summary_fig_filename = args.summaryfig
    num_folds = args.numfolds

    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Model-building for dataset file " + dataset_filename + " beginning at " + current_time)

    # Load dataset.
    X, y = load_dataset(dataset_filename, target_column, pred_start_column, pred_end_column) # Remember that with Python, numbering begins at 0.

    # Get model list.
    models = define_models()

    ''' 
    As of 2021-11-06, we are not scaling predictors. I think it might affect individual predictions using the trained model. ...
    # Scale predictor values.
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    # Why we don't need to scale target values: https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X_scaled = scaling.transform(X)
    X_scaled = pandas.DataFrame(X_scaled) # This part necessary because scaling transforms it into a numpy ndarray, and later functions require a Pandas DataFrame as input. 
    '''
    
    # Evaluate models.
    results_raw = evaluate_models(X = X, y = y, models = models, folds = num_folds, metric = 'accuracy') # num_folds default is 3. 

    # Summarize results.
    #summarize_results(results, results_summary_fig_filename)

    # Save raw results.
    results_raw.to_csv(raw_results_filename)
    #results_raw_by_group.to_csv(raw_results_by_group_filename)

    # Mark time at finish of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Model-building concluding at " + current_time)

if __name__ == "__main__":
    main()