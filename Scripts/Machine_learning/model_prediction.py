#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:31:48 2020

@author: alonzowolf
"""

import argparse
import pandas                              
from sklearn.ensemble import ExtraTreesRegressor
from datetime import datetime
from datetime import date
import pickle

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

# Set variables.
'''
target = "PercentGrowth"
descriptor_type = 'PaDEL'
gene_subset_type = 'L1000'
preprocessing_method = 'YuGene'
id_type = 'Entrez'
dataset = 'ONeil'
target_variable_type = 'pg'
'''
parser = argparse.ArgumentParser(description='Arguments for model_prediction.py.')
parser.add_argument("--target", 
                    default="PercentGrowth", 
                    choices=["PercentGrowth", "BestComboScore"],
                    type=str, 
                    help="The name of the target variable, e.g. 'PercentGrowth' or 'BestComboScore'.")
parser.add_argument("--dt", 
                    default="PaDEL",
                    choices=["PaDEL", "alvaDesc"],
                    type=str, 
                    help="The name of the chemical descriptor type used, e.g. 'PaDEL' or 'alvaDesc'.")
parser.add_argument("--gst", 
                    default="L1000", 
                    type=str, 
                    help="The name of the gene subset type used, e.g. 'L1000'.")
parser.add_argument("--pm", 
                    default="YuGene",
                    choices=["YuGene", "SCAN", "UPC"],
                    type=str, 
                    help="The name of the gene-expression normalization method used, e.g. 'YuGene' or 'SCAN'.")
parser.add_argument("--id", 
                    default="Entrez",
                    choices=["Entrez", "HUGO", "Ensembl"],
                    type=str, 
                    help="The name of the gene annotation system used, e.g. 'HUGO' or 'Entrez'.")
parser.add_argument("--dataset", 
                    default="GEO",
                    choices=["ONeil", "GEO", "Gao"],
                    type=str, 
                    help="The name of the data set used, e.g. 'ONeil' or 'GEO'.")

args = parser.parse_args()
target = args.target
descriptor_type = args.dt
gene_subset_type = args.gst
preprocessing_method = args.pm
id_type = args.id
dataset = args.dataset
if dataset == "GEO":
    dataset_folder = "NCBI-GEO"
else:
    dataset_folder = dataset

# Establish connection to the log file. 
today = date.today()
date = today.strftime("%b-%d-%Y")
log_file_name = "/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Results/log_test_set_scoring_" + date + ".txt"
log_file = open(log_file_name, "a")

# Mark time at beginning of model-building. 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print('', file = log_file)
print("Test-set scoring with the following variables beginning at " + current_time + ": " + target + ", " + descriptor_type + ", " + gene_subset_type + ", " + preprocessing_method + ", " + id_type + ", " + dataset + ".", file = log_file)
log_file.close()

# Load the model from disk.
model_type = 'ExtraTreesRegressor'
filename = '/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Data/finalized_' + model_type + '_' + target + '_model_' + descriptor_type + '_' + gene_subset_type + '_' + preprocessing_method + '_' + id_type + '.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the test data.
dataset_filename = '/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Data/' + dataset_folder + '/' + dataset + '_test_data_' + descriptor_type + '_' + gene_subset_type + '_' + preprocessing_method + '_' + id_type + '_cs_pg.csv'
target_column = 4
X_test, Y_test, groups = load_dataset(dataset_filename, target_column, 1, 5, -99)

# Make predictions.
result = pandas.Series(loaded_model.predict(X_test))
result_filename = '/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Results/Cell_growth/' + dataset + '_scored_' + descriptor_type + '_' + gene_subset_type + '_' + preprocessing_method + '_' + id_type + '_' + target + '.csv'
result.to_csv(result_filename)

# Mark time at end of model-building.
# Mark time at beginning of model-building. 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
log_file = open(log_file_name, "a")
print('', file = log_file)
print("Test-set scoring concluding at " + current_time, file = log_file)
log_file.close()