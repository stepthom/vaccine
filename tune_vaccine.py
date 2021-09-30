import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import numpy as np
import pandas as pd
import os
import time
import socket

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import scipy.stats

import optuna

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--study-name', type=str, default="seasonal_no_es")
    parser.add_argument('--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('--n-trials', type=int, default=5)
    args = parser.parse_args()
    
    def objective(trial, study_name, X, y, cat_cols):
       
        # Same as FLAML's model.py (LGBMEstimator::search_space())
        upper = 32768
        params = {
              "num_leaves": trial.suggest_int("num_leaves", 4, upper, log=True),
              "min_child_samples": trial.suggest_int("min_child_samples", 2, 2 ** 7 + 1, log=True),
              "learning_rate": trial.suggest_float("learning_rate", 1 / 1024, 1.0, log=True),
              "max_bin": trial.suggest_int("max_bin", 2**3, 2**11, log=True),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
              "reg_alpha":  trial.suggest_float("reg_alpha", 1/1024, 1024, log=True),
              "reg_lambda": trial.suggest_float("reg_lambda", 1/1024, 1024, log=True),
              "n_jobs": 5,
              "verbosity": -1,
              "seed": 77,
        }
        
        fit_params= {
              'feature_name': "auto",
              'categorical_feature': cat_cols,
        }
        
        if study_name == "seasonal_no_es":
            params["n_estimators"] = trial.suggest_int("n_estimators", 4, upper, log=True)
        elif study_name == "seasonal_yes_es":
            params["n_estimators"] =  upper
      
        cv_scores = []
        best_iterations = []

        start = time.time()
        num_cv = 15
        skf = StratifiedKFold(n_splits=num_cv, random_state=42, shuffle=True)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.loc[train_index].reset_index(drop=True), X.loc[val_index].reset_index(drop=True)
            y_train, y_val = y[train_index], y[val_index]
            
            if study_name == "seasonal_yes_es":
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['early_stopping_rounds'] = 100
                fit_params['verbose'] = 200


            estimator = LGBMClassifier(**params)
            estimator.fit(X_train, y_train, **fit_params)

            y_val_pred_proba = estimator.predict_proba(X_val)
            cv_scores.append(roc_auc_score(y_val, y_val_pred_proba[:,1]))
            best_iterations.append(estimator.best_iteration_)
            
        duration = time.time() - start
            
        print("CV scores:")
        print(cv_scores)
        
        # pessimistic case
        score = np.mean(cv_scores) - np.std(cv_scores)
       
        # Log for later
        trial.set_user_attr("estimator_params", params)
        trial.set_user_attr("cv_scores", cv_scores)
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("duration", duration)
        trial.set_user_attr("hostname", socket.gethostname())
        
        return score
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler= optuna.samplers.TPESampler(
            n_startup_trials = 100,
            n_ei_candidates = 10,
            constant_liar=True,
        ),
        direction="maximize",
        load_if_exists = True,
    )
    
    train_fn = "data/vaccine_seasonal_train.csv"
    id_col = "respondent_id"
    target_col = "seasonal_vaccine"

    train_df = pd.read_csv(train_fn)

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)
    
    cat_cols = [ 
        "age_group", "education", "race", "sex", "income_poverty", "marital_status",
        "rent_or_own", "employment_status", "hhs_geo_region", "census_msa", "employment_industry", "employment_occupation" ]
    
    X[cat_cols] = X[cat_cols].astype('category')
    for cat_col in cat_cols:
        X[cat_col] = X[cat_col].cat.add_categories('__NAN__')
    X[cat_cols] = X[cat_cols].fillna('__NAN__')
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    
    study.optimize(lambda trial: objective(trial, args.study_name, X, y, cat_cols),
                    n_trials=args.n_trials,  gc_after_trial=True)
    return
    
if __name__ == "__main__":
    main()
