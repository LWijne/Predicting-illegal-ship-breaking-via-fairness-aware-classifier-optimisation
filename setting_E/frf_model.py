############################# FAIR RANDOM FOREST #############################

#!/usr/bin/env python
# coding: utf-8

# Import essential packages
import pandas as pd
import numpy as np
import sys
import copy

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# Fair Random Forest
from fair_trees import FairRandomForestClassifier

# HyperOpt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Path
sys.path.append('../')

# No warnings
pd.options.mode.chained_assignment = None

from warnings import filterwarnings
filterwarnings('ignore')

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

def cross_val_score_custom(model, X, y, cv=10):
    '''
    Evaluate the ROC AUC score by cross-validation.

            Parameters:
                    model (FairRandomForestClassifier object): The model.
                    X (array-like): The training data.
                    y (array-like): The labels.
                    cv (int): Number of folds.

            Returns:
                    roc_auc (float): The ROC AUC score.
    '''
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    auc_list = []
    
    s = X[sensitive_col]
    splitter_y = y.astype(str) + s.astype(str)

    # Looping over the folds
    for trainset, testset in splitter.split(X,splitter_y):

        # Splitting and reparing the data, targets and sensitive attributes
        X_train = X[X.index.isin(trainset)]
        y_train = y[y.index.isin(trainset)]
        
        X_test = X[X.index.isin(testset)]
        y_test = y[y.index.isin(testset)]
        
        s_train = pd.DataFrame(X_train[sensitive_col]).values.astype(int)
        
        X_train = X_train.drop(columns=[sensitive_col])
        X_test = X_test.drop(columns=[sensitive_col])
        
        X_train = ct.fit_transform(X_train)
        X_test = ct.transform(X_test)

        # Initializing and fitting the classifier
        clf = copy.deepcopy(model)
        clf.fit(X_train, y_train, s_train)

        # Final predictions
        y_pred_probs = clf.predict_proba(X_test).T[1]
        y_true = y_test

        auc_list.append(roc_auc_score(y_true,y_pred_probs))


    # Final results
    auc_list = np.array(auc_list)
    roc_auc = np.nanmean(auc_list, axis=0)
    return roc_auc

def best_model(trials):
    '''
    Retrieve the best model.

            Parameters:
                    trials (Trials object): Trials object.

            Returns:
                    trained_model (FairRandomForestClassifier object): The best model.
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
    model = FairRandomForestClassifier(
      random_state=random_state,
      theta=params['theta'],
      n_bins=params['n_bins'],
      max_depth=params['max_depth'],
      bootstrap=params['bootstrap'],
      n_estimators=params['n_estimators'],
      min_samples_split=params['min_samples_split'],
      min_samples_leaf=params['min_samples_leaf'],
      max_features=params['max_features']
    )
    roc_auc = cross_val_score_custom(
      model,
      X_train_df,
      y_train_df,
      cv=K
    )

    return {'loss': -roc_auc, 'status': STATUS_OK, 'trained_model': model}


############################# Training the classifier, predictions and outcomes #############################

def fair_random_forest_(th):
    '''
    Computes the average and std of AUC and SDP over K folds.

            Parameters:
                    th (float): The theta value for FRF.

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
        
        # Splitting and preparing the data, targets and sensitive attributes
        X_train_df = sloopschepen["X"][sloopschepen["X"].index.isin(trainset)]
        y_train_df = sloopschepen["y"][sloopschepen["y"].index.isin(trainset)]
        
        X_test_df = sloopschepen["X"][sloopschepen["X"].index.isin(testset)]
        y_test_df = sloopschepen["y"][sloopschepen["y"].index.isin(testset)]

        params = {
            'theta': hp.choice('theta', [th]),
            'n_bins': hp.choice('n_bins', [256]),
            'bootstrap': hp.choice('bootstrap', [True]),
            'max_depth': hp.uniformint('max_depth', 1, 20, q=1.0),
            'max_features': hp.uniform("max_features", 0.05, 0.95),
            'n_estimators': hp.uniformint('n_estimators', 100, 500, q=1.0),
            'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 10, q=1.0),
            'min_samples_split': hp.uniformint('min_samples_split', 2, 20, q=1.0),
        }

        trials = Trials()

        opt = fmin(
            fn=objective,
            space=params,
            algo=tpe.suggest,
            max_evals=hyperopt_evals,
            trials=trials
        )

        # Initializing and fitting the classifier
        cv = best_model(trials)
        
        s_train = pd.DataFrame(X_train_df[sensitive_col]).values.astype(int)
        s_test = X_test_df[sensitive_col]
        
        X_train_df = X_train_df.drop(columns=[sensitive_col])
        X_test_df = X_test_df.drop(columns=[sensitive_col])
        
        X_train_df = pd.DataFrame(ct.fit_transform(X_train_df))
        X_test_df = pd.DataFrame(ct.transform(X_test_df))
        
        cv.fit(X_train_df, y_train_df, s_train)

        # Final predictions
        y_pred_probs = cv.predict_proba(X_test_df).T[1]
        y_true = y_test_df

        mean_roc_auc.append(roc_auc_score(y_true, y_pred_probs))
        mean_strong_dp.append(strong_demographic_parity_score(s_test, y_pred_probs))
    
    return np.average(mean_roc_auc), np.average(mean_strong_dp), np.std(mean_roc_auc), np.std(mean_strong_dp)

auc_list = []
sdp_list = []
std_auc_list = []
std_sdp_list = []

theta_list = np.arange(0.0, 1.1, 0.1)

for th in theta_list:
    roc_auc, strong_dp, std_auc, std_sdp = fair_random_forest_(th)
    auc_list.append(roc_auc)
    sdp_list.append(strong_dp)
    std_auc_list.append(std_auc)
    std_sdp_list.append(std_sdp)
    print(((th*10+1)/11)*100, "% complete")

############################# Results #############################

print("auc_frf_setD =", auc_list)
print("sdp_frf_setD =", sdp_list)
print("std_auc_frf_setD =", std_auc_list)
print("std_sdp_frf_setD =", std_sdp_list)
