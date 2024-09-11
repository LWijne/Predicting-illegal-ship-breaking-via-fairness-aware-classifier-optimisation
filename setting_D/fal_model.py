############################# FAIR ADVERSARIAL LEARNING #############################

#!/usr/bin/env python
# coding: utf-8

# Import essential packages
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import copy
import math

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# AIF360
from aif360.sklearn.inprocessing import AdversarialDebiasing

# HyperOpt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Path
sys.path.append('../')

# No warnings
pd.options.mode.chained_assignment = None

from warnings import filterwarnings
filterwarnings('ignore')

# Disable eager execution for Adversarial Debiasing
tf.compat.v1.disable_eager_execution()

############################# Data pre-processing and feature selection functions #############################

def read_data():
    '''
    Reads the file and filters the relevant information.

            Parameters:


            Returns:
                    sloopschepen (pandas.DataFrame): DataFrame containing the relevant data.
    '''

    # Get the file
    sb_file = "sloopschepen_2024-01-22.csv"
    # Read
    sloopschepen = pd.read_csv(sb_file).drop("Unnamed: 0",axis=1)
    # Filter out active ships
    sloopschepen = sloopschepen[sloopschepen.dismantled == 1]
    # Get the relevant columns
    sloopschepen = sloopschepen[predictors+[target_col]].reset_index(drop=True)
    return sloopschepen

class MissIndicator():
    
    def __init__(self):
        self.is_fit = False
        
    def fit(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        
    def transform(self, X, y=None):
        return np.concatenate([X, self.mi.transform(X)], axis=1)
    
    def fit_transform(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        return np.concatenate([X, self.mi.transform(X)], axis=1)


class Clamper():
    
    def __init__(self):
        self.is_fit = False
        self.values_to_keep = {}
        
    def _get_values_to_keep_from_value_counts(self, value_counts):
        values = value_counts.keys()
        counts = value_counts.values.astype(int)
        count_p = counts / sum(counts)
        min_p_increase = 1/len(values)
        index_to_keep = np.argmin(abs(count_p - min_p_increase))
        values_to_keep = values[:index_to_keep]

        return values_to_keep
    
    def fit_transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
            transformed_X[column].loc[
                ~(transformed_X[column].isin(self.values_to_keep[column]))
            ] = "other"
        self.is_fit = True
        return transformed_X
    
    def fit(self, X, y=None):
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
        self.is_fit = True
        
    def transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            transformed_X[column].loc[
                ~(transformed_X[column].isin(self.values_to_keep[column]))
            ] = "other"
        
        return transformed_X

def data_pre_processing(sloopschepen):
    '''
    Missing value imputation and converting the sensitive attribute into a binary attribute.

            Parameters:
                    sloopschepen (pandas.DataFrame): DataFrame containing the data.

            Returns:
                    sloopschepen (pandas.DataFrame): DataFrame containing the preprocessed data.
    '''

    EOL_FOC_list = ["KNA", "COM", "PLW", "TUV", "TGO", "TZA", "VCT", "SLE"]
    for x in ["country_current_flag", "country_previous_flag"]:
        sloopschepen[x][~sloopschepen[x].isin(EOL_FOC_list)] = 0 # non-FOC for ship-breaking
        sloopschepen[x][sloopschepen[x].isin(EOL_FOC_list)] = 1 # FOC for ship-breaking acc to NGO SP
        
    # Replace NaN's with 'missing' for string columns
    for x in cat_columns:
        sloopschepen[x] = sloopschepen[x].fillna('missing')
        # Also replace values with "unknown" or similar to missing
        sloopschepen[x][sloopschepen[x].apply(str.lower).str.contains("unknown|unspecified")] = 'missing'
        sloopschepen[x][sloopschepen[x].apply(str.lower) == "unk"] = 'missing' 

    return sloopschepen


def data_prep(df, K, predictors, target_col):
    '''
    Prepares a dictionary of X, y and folds.

            Parameters:
                    df (pandas.DataFrame): DataFrame containing the data.
                    K (int): Number of cross validation folds.
                    predictors (list): List of predictor columns.
                    target_col (str): The target column.

            Returns:
                    data_prep_dict (dict): Dictionary with X, y and folds.
    '''
    # Select targets from development data
    targets = df[target_col].reset_index(drop=True)
    
    # Select predictors from data
    df = df[predictors].reset_index(drop=True)
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    
    # Create result dictionary
    data_prep_dict = {}
    data_prep_dict["X"] = df
    data_prep_dict["y"] = targets
    data_prep_dict['folds'] = splitter
    
    return data_prep_dict  


############################# Parameters #############################

K = 10 # K-fold CV

hyperopt_evals = 100 # Max number of evaluations for HPO

target_col = "beached" # Target

sensitive_col = "country_current_flag" # Sensitive attribute

random_state = 42 # Seed to be used for reproducibility

# Define list of predictors to use
predictors = [
    "vessel_type",
    "gross_tonnage",
    "port_of_registry",
    "country_current_flag",
    "country_previous_flag",
    "years_since_final_flag_swap",
    "pop_current_flag",
    "gdpcap_current_flag",
    "speed",
    "age_in_months"
]

# Specify which predictors are numerical
num_columns = [
    "gross_tonnage",
    "years_since_final_flag_swap",
    "speed",
    "age_in_months",
    "pop_current_flag",
    "gdpcap_current_flag"
]

# Specify which predictors are categorical and need to be one-hot-encoded
cat_columns = [
    "vessel_type",
    "port_of_registry"
]

num_transformer = Pipeline([
    ('scaler', RobustScaler()),
    ("mindic", MissIndicator()),
    ('imputer', SimpleImputer())
])
cat_transformer = Pipeline([
    ("clamper", Clamper()),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

ct = ColumnTransformer([
    ('num_transformer', num_transformer, num_columns),
    ('cat_transformer', cat_transformer, cat_columns)
],
    remainder='passthrough'
)

sloopschepen = read_data()
sloopschepen = data_pre_processing(sloopschepen)

# Prepare the data 
sloopschepen = data_prep(df=sloopschepen,
                   K=K,
                   predictors=predictors,
                   target_col=target_col)


def strong_demographic_parity_score(s, y_prob):
    '''
    Returns the strong demographic parity score.

            Parameters:
                    s (array-like): The sensitive features over which strong demographic parity should be assessed.
                    y_prob (array-like): The predicted probabilities returned by the classifier.

            Returns:
                    sdp (float): The strong demographic parity score.
    '''
    y_prob = np.array(y_prob)
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    
    sensitive_aucs = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            sensitive_aucs.append(1) 
        else:
            sens_auc = 0
            for s_unique in np.unique(s[:, s_column]):
                s_bool = (s[:, s_column]==s_unique)
                auc = roc_auc_score(s_bool, y_prob)
                auc = max(1-auc, auc)
                sens_auc = max(sens_auc, auc)
            sensitive_aucs.append(sens_auc)
    
    s_auc = sensitive_aucs[0] if len(sensitive_aucs)==1 else sensitive_aucs
    sdp = abs(2*s_auc-1)
    return sdp


############################# HPO #############################


def cross_val_score_custom(params, X, y, cv=10):
    '''
    Evaluate the ROC AUC score by cross-validation.

            Parameters:
                    params (dict): The parameters for the AdversarialDebiasing model.
                    X (array-like): The training data.
                    y (array-like): The labels.
                    cv (int): Number of folds.

            Returns:
                    auc_perf (float): The ROC AUC score of the predictions and the labels.
                    auc_fair (float): The ROC AUC score of the predictions and the sensitive attribute.
    '''
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    auc_perf_list = []
    auc_fair_list = []
    
    s = X[sensitive_col]
    splitter_y = y.astype(str) + s.astype(str)

    # Looping over the folds
    for trainset, testset in splitter.split(X,splitter_y):

        # Splitting and reparing the data, targets and sensitive attributes
        X_train = X[X.index.isin(trainset)]
        y_train = y[y.index.isin(trainset)]
        
        X_test = X[X.index.isin(testset)]
        y_test = y[y.index.isin(testset)]
        
        s_train = X_train[sensitive_col]
        s_test = X_test[sensitive_col].astype(int)
        
        X_train = X_train.drop(columns=[sensitive_col])
        X_test = X_test.drop(columns=[sensitive_col])
        
        X_train = pd.DataFrame(ct.fit_transform(X_train))
        X_test = pd.DataFrame(ct.transform(X_test))

        # Initializing and fitting the classifier
        cv = AdversarialDebiasing(
                  prot_attr=s_train,
                  debias=True,
                  random_state=random_state,
                  adversary_loss_weight=params['adversary_loss_weight'],
                  num_epochs=params['num_epochs'],
                  batch_size=params['batch_size'],
                  classifier_num_hidden_units=params['classifier_num_hidden_units']
              )
        cv.fit(X_train, y_train)

        # Final predictions
        y_pred_probs = cv.predict_proba(X_test).T[1]
        y_true = y_test
        
        nan_inf_error_list = [math.isfinite(x) for x in y_pred_probs]
        if 0.0 in nan_inf_error_list:
            print("Error")
            auc_perf_list.append(0.5)
            auc_fair_list.append(0.5)
        else:
            auc_perf_list.append(roc_auc_score(y_true,y_pred_probs))
            auc_fair_list.append(0.5 + abs(0.5 - roc_auc_score(s_test, y_pred_probs)))


    # Final results
    auc_perf_list = np.array(auc_perf_list)
    auc_perf = np.nanmean(auc_perf_list, axis=0)
    auc_fair_list = np.array(auc_fair_list)
    auc_fair = np.nanmean(auc_fair_list, axis=0)
    return auc_perf, auc_fair

def best_model(trials):
    '''
    Retrieve the best model.

            Parameters:
                    trials (Trials object): Trials object.

            Returns:
                    trained_model (dict): The parameters for the best model.
    '''
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    trained_model = best_trial_obj['result']['trained_model']
    return trained_model

def objective(params):
    '''
    Retrieve the loss for a model created by certain parameters.

            Parameters:
                    params (dict): The parameters to create the model.

            Returns:
                    (dict): The loss, status and trained model.
    '''
    roc_auc_y, roc_auc_s = cross_val_score_custom(
      params,
      X_train_df,
      y_train_df,
      cv=10
    )
    goal = (1-theta) * roc_auc_y - theta * roc_auc_s
    
    return {'loss': -goal, 'status': STATUS_OK, 'trained_model': params}

############################# Training the classifier, predictions and outcomes #############################

def fair_adversarial_learning_(th):
    '''
    Computes the average and std of AUC and SDP over K folds.

            Parameters:
                    th (float): The theta value for FAL.

            Returns:
                    roc_auc (float): The average of the ROC AUC list.
                    strong_dp (float): The average of the strong demographic parity list.
                    std_auc (float): The standard deviation of the ROC AUC list.
                    std_sdp (float): The standard deviation of the strong demographic parity list.
    '''

    mean_roc_auc = []
    mean_strong_dp = []
    
    y = sloopschepen["y"]
    s = sloopschepen["X"][sensitive_col]
    splitter_y = y.astype(str) + s.astype(str)

    # Looping over the folds
    for trainset, testset in sloopschepen["folds"].split(sloopschepen["X"],splitter_y):
        
        global X_train_df
        global y_train_df
        global theta
        theta = th

        # Splitting and preparing the data, targets and sensitive attributes
        X_train_df = sloopschepen["X"][sloopschepen["X"].index.isin(trainset)]
        y_train_df = sloopschepen["y"][sloopschepen["y"].index.isin(trainset)]
        X_test_df = sloopschepen["X"][sloopschepen["X"].index.isin(testset)]
        y_test_df = sloopschepen["y"][sloopschepen["y"].index.isin(testset)]
        
        params = {
            'adversary_loss_weight': hp.uniform('adversary_loss_weight', 0.0, 1.0),
            'num_epochs': hp.uniformint('num_epochs', 50, 500, q=1.0),
            'batch_size': hp.uniformint('batch_size', 16, 1024, q=1.0),
            'classifier_num_hidden_units': hp.uniformint('classifier_num_hidden_units', 40, 1000, q=1.0)
        }

        trials = Trials()

        opt = fmin(
            fn=objective,
            space=params,
            algo=tpe.suggest,
            max_evals=hyperopt_evals,
            trials=trials
        )

        params_best_model = best_model(trials)
    
        s_train = X_train_df[sensitive_col]
        s_test = X_test_df[sensitive_col]
        X_train_df = X_train_df.drop(columns=[sensitive_col])
        X_test_df = X_test_df.drop(columns=[sensitive_col])
        
        X_train_df = pd.DataFrame(ct.fit_transform(X_train_df))
        X_test_df = pd.DataFrame(ct.transform(X_test_df))

        # Initializing and fitting the classifier
        cv = AdversarialDebiasing(
                  prot_attr=s_train,
                  debias=True,
                  random_state=random_state,
                  adversary_loss_weight=params_best_model['adversary_loss_weight'],
                  num_epochs=params_best_model['num_epochs'],
                  batch_size=params_best_model['batch_size'],
                  classifier_num_hidden_units=params_best_model['classifier_num_hidden_units']
              )
        cv.fit(X_train_df, y_train_df)

        # Final predictions
        y_pred_probs = cv.predict_proba(X_test_df).T[1]
        y_true = y_test_df

        nan_inf_error_list = [math.isfinite(x) for x in y_pred_probs]
        if 0.0 in nan_inf_error_list:
            print("Error")
            mean_roc_auc.append(0.5)
            mean_strong_dp.append(0.5)
        else:
            mean_roc_auc.append(roc_auc_score(y_true, y_pred_probs))
            mean_strong_dp.append(strong_demographic_parity_score(s_test, y_pred_probs))

    return np.average(mean_roc_auc), np.average(mean_strong_dp), np.std(mean_roc_auc), np.std(mean_strong_dp)

auc_list = []
sdp_list = []
std_auc_list = []
std_sdp_list = []

theta_list = np.arange(0.0, 1.1, 0.1)

for th in theta_list:
    roc_auc, strong_dp, std_auc, std_sdp = fair_adversarial_learning_(th)
    auc_list.append(roc_auc)
    sdp_list.append(strong_dp)
    std_auc_list.append(std_auc)
    std_sdp_list.append(std_sdp)
    print(((th*10+1)/11)*100, "% complete")

############################# Results #############################

print("auc_fal_setD =", auc_list)
print("sdp_fal_setD =", sdp_list)
print("std_auc_fal_setD =", std_auc_list)
print("std_sdp_fal_setD =", std_sdp_list)
