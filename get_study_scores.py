import optuna
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--optuna-storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
parser.add_argument('-s', '--save-results', type=int, default=0)
args = parser.parse_args()

study_names = ['seasonal_no_es', 'seasonal_yes_es']
for study_name in study_names:
    print("===================================")
    print("Study: {}".format(study_name))
    print("")
    
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)

    df = study.trials_dataframe()
    print(df['state'].value_counts())
   
    if study_name == "seasonal_no_es":
        df['best_iterations'] = '-'
    else:
        df['params_n_estimators'] = '-'
        df['best_iterations'] = df['user_attrs_best_iterations'].apply(np.mean)
        
        
    
    
    print("Best:")
    df = df.sort_values('value', ascending=False)
    col_list = [
        'number', 
        'value', 
        'params_n_estimators',
        'params_num_leaves',
        'params_min_child_samples',
        'params_learning_rate',
        'params_max_bin',
        'params_colsample_bytree',
        'params_reg_alpha',
        'params_reg_lambda',
        'duration', 
        'best_iterations', 
        #'user_attrs_hostname',
        'state']
    print(df[col_list].head(20))
    if args.save_results == 1:
        df.to_csv('out/{}.csv'.format(study_name), index=False)
