#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:54:45 2021

@author: alonzowolf
"""

import argparse
import sys
import random
import warnings
import pandas                 
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                  
from numpy import mean
from numpy import std
#from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
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
def load_dataset(filepath,predstartcol, predendcol=-99):
    data = pandas.read_csv(filepath)
    
    if predendcol==-99:
        X = data.iloc[:,predstartcol:len(data.columns)]
    else:
        X = data.iloc[:,predstartcol:predendcol]
    
    return X

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
    pred_start_column = ''
    model_filename = ''
    predictions_filename = ''
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dataset',
                    required=True,
                    type=str, 
                    dest="dataset",
                    metavar="<path to data set>",
                    help="Full path to the data set" )
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
        parser.add_argument('-m', '--model',
                    required=True,
                    type=str,
                    dest="model",
                    metavar="<model filename>",
                    help="Path to the saved model" )
        parser.add_argument('-p', '--predictions',
                    required=True,
                    type=str, 
                    dest="predictions",
                    metavar="<predictions filename>",
                    help="Path to save the predictions to" )
        args = parser.parse_args()
        
    except argparse.ArgumentError:
        print('scoring.py -d <dataset> -s <predstartcol> -m <model> -p <predictions>')
        sys.exit(2)
    
    dataset_filename = args.dataset
    pred_start_column = args.predstartcol
    pred_end_column = args.predendcol
    model_filename = args.model
    predictions_filename = args.predictions

    # Mark time at beginning of scoring. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("Scoring for dataset file " + dataset_filename + " beginning at " + current_time)

    # Load dataset.
    X = load_dataset(dataset_filename, pred_start_column, pred_end_column) # Remember that with Python, numbering begins at 0.
    
    ''' 
    As of 2021-11-06, we are not scaling predictors. I think it might affect individual predictions using the trained model. ...
    # Scale predictor values.
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    # Why we don't need to scale target values: https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X_scaled = scaling.transform(X)
    X_scaled = pandas.DataFrame(X_scaled) # This part necessary because scaling transforms it into a numpy ndarray, and later functions require a Pandas DataFrame as input. 
    '''
    
    loaded_model = pickle.load(open(model_filename, 'rb'))
    result = pandas.Series(loaded_model.predict(X))
    result.to_csv(predictions_filename)
    print(result)
    
    # Mark time at end of model-building.
    # Mark time at beginning of model-building. 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print("Scoring concluding at " + current_time)

if __name__ == "__main__":
    main()