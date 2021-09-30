import optuna
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
args = parser.parse_args()

study_names = ['seasonal_no_es', 'seasonal_yes_es']
for study_name in study_names:
    print("Deleting study: {}".format(study_name))
    optuna.delete_study(study_name=study_name, storage=args.storage)
   