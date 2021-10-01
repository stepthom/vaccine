import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import numpy as np
import pandas as pd
import os
import time
import json

from flaml import tune

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import scipy.stats


# Helper functions to write JSON to disk
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_json(fn, json_obj):
    # Write json_obj to a file named fn

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--early-stopping', type=int, default=0)
    parser.add_argument('--time-budget', type=int, default=30)
    args = parser.parse_args()
    
    def objective(params):
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
            "rent_or_own", "employment_status", "hhs_geo_region", "census_msa", "employment_industry", "employment_occupation" 
        ]

        X[cat_cols] = X[cat_cols].astype('category')
        for cat_col in cat_cols:
            X[cat_col] = X[cat_col].cat.add_categories('__NAN__')
        X[cat_cols] = X[cat_cols].fillna('__NAN__')
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        
        fit_params= {
              'feature_name': "auto",
              'categorical_feature': cat_cols,
        }
        
        cv_scores = []
        best_iterations = []

        start = time.time()
        num_cv = 15
        skf = StratifiedKFold(n_splits=num_cv, random_state=42, shuffle=True)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.loc[train_index].reset_index(drop=True), X.loc[val_index].reset_index(drop=True)
            y_train, y_val = y[train_index], y[val_index]
            
            if args.early_stopping == 1:
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['early_stopping_rounds'] = 100
                fit_params['verbose'] = 0

            estimator = LGBMClassifier(**params)
            estimator.fit(X_train, y_train, **fit_params)

            y_val_pred_proba = estimator.predict_proba(X_val)
            cv_scores.append(roc_auc_score(y_val, y_val_pred_proba[:,1]))
            best_iterations.append(estimator.best_iteration_)
            
        duration = time.time() - start
        
        # pessimistic case
        score = np.mean(cv_scores) - np.std(cv_scores)
        tune.report(score=score, duration=duration, cv_scores=cv_scores, best_iterations=best_iterations) 
        
        print("params: {}".format(params))
        print("score: {}".format(score))
        
    upper = 2**15    
    config={
        "num_leaves":  tune.lograndint(lower=4, upper=upper),
        "min_child_samples":  tune.lograndint(lower=2, upper=2**7 + 1),
        "learning_rate":  tune.loguniform(lower=1 / 1024, upper=1.0),
        "max_bin":  tune.lograndint(lower=2**3, upper=2**11),
        "colsample_bytree": tune.uniform(lower=0.01, upper=1.0),
        "reg_alpha": tune.loguniform(lower=1 / 1024, upper=1024),
        "reg_lambda": tune.loguniform(lower=1 / 1024, upper=1024),
    }
    low_cost_partial_config={
        'num_leaves': 4,
        'min_child_samples': 20,
        'learning_rate': 1.0,
        'max_bin': 8,
        'colsample_bytree': 0.1,
        "reg_alpha": 1/1024,
        "reg_lambda": 1.0,
    }
    if args.early_stopping == 1:
        config["n_estimators"] = upper
    else:
        config["n_estimators"] = tune.lograndint(lower=4, upper=upper)
        low_cost_partial_config["n_estimators"] = 4

    analysis = tune.run(
        objective,    # the function to evaluate a config
        config=config,
        low_cost_partial_config=low_cost_partial_config,
        metric='score',
        mode='max',
        num_samples=-1,
        time_budget_s=args.time_budget, 
        local_dir='logs/',
        # verbose=0,          # verbosity    
        # use_ray=True, # uncomment when performing parallel tuning using ray
    )

    dump_json("logs/{}_all_results.json".format(args.early_stopping), analysis.results) 
    dump_json("logs/{}_best_trial.json".format(args.early_stopping), analysis.best_trial.last_result) 
    dump_json("logs/{}_best_config.json".format(args.early_stopping), analysis.best_config) 
    df = pd.DataFrame.from_dict(analysis.results).T
    df.to_csv('logs/{}_all_results.csv'.format(args.early_stopping))
    
    
if __name__ == "__main__":
    main()
