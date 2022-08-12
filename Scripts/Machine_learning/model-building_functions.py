# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:27:24 2020

@author: Lon Fong
"""

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
def load_dataset(filepath, predstartcol, predendcol=-99, targetcol, groupscol):
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

	models['xgb'] = xgb.XGBRegressor()

	print('Defined %d models' % len(models))
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
            
        dataset_split.append(indices) # Append() here because we want dataset_split to be a list of lists. Extend() above (fold.extend()) because we want fold to be a simple vector (list) of indices without any kind of separation
        
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
def evaluate_models(X, y, models, groups, folds, scoring_function, greater_is_better):
    results = dict()
    for name, model in models.items():
        # Evaluate the model.
        scores = robust_evaluate_model(X, y, model, groups, folds, scoring_function, greater_is_better)
        # Show process.
        if scores is not None:
            # Store a result.
            results[name] = scores
            mean_score, std_score = mean(scores), std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
        else:
            print('>%s: error' % name)
    return results

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
def robust_evaluate_model(X, y, model, groups, folds, scoring_function, greater_is_better): # For robust evaluation of models; traps exceptions and ignores warnings.
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, groups, folds, scoring_function, greater_is_better)
    except:
        scores = None
    return scores

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
def evaluate_model(X, y, model, groups, folds, scoring_function, greater_is_better):
    # Split training/test. 
    split = cv_split_by_groups(groups, folds)
    
    # Loop through each fold and build the model, make predictions, and then score the model. 
    fold_scores = list()
    all_indices = range(len(X))
    for i in range(len(split)): 
        test_indices = split[i]
        train_indices = all_indices.difference(test_indices)
        X_train = X.iloc[train_indices,:]
        y_train = y.iloc[train_indices,:]
        X_test = X.iloc[test_indices,:]
        y_test = y.iloc[test_indices,:]
        
        # Build model. 
        model.fit(X_train, y_train) # Do only scikit-learn models have the .fit() method?
        
        # Use model to predict. 
        predicted = model.predict(X_test)
        
        # Evaluate model.
        mean_score, SD_score = score_model(predicted, y_test, groups, scoring_function, greater_is_better)
        
        # Add mean_score (mean of scores of the individual groups in fold i) to fold_scores.
        fold_scores.extend(mean_score)
    
    return mean_score, SD_score
        
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
def score_model(y_true, y_pred, groups, scoring_function, greater_is_better = True):    
    group_scores = list()
    # For each group, score the performance of the model using the scoring function defined by scoring_function. 
    unique_groups = list(set(groups))
    for i in unique_groups:
        # Get the indices of y_true, y_pred that correspond to group i. 
        indices = [j for j, x in enumerate(groups) if x == i]
        # Subset y_true, y_pred to include only entries in group i. 
        y_true_i = y_true[indices] # Can we do it this way? Yes, if this is a series object of pandas.core.series module. If this is a dataframe, we will need to use .iloc[:,:], of course.
        y_pred_i = y_pred[indices]
        
        # Score and add the score to the scores list.
        group_scores.extend(scoring_function(y_true_i, y_pred_i))
    
    mean_score = stats.mean(group_scores)
    # SD_score = stats.stdev(group_scores)
    
    return mean_score # mean_score is the average score across the groups.

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
def summarize_results(results, maximize = True, top_n = 10, fig_path):
    # Check for no results.
    if len(results)==0:
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
    pyplot.savefig(fig_path + 'spotcheck.png')



