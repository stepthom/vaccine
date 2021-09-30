#!/bin/bash
#SBATCH --job-name=SeasonalSearcher
#SBATCH --cpus-per-task=5
#SBATCH --mem=20gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/SeasonalSearcher-%x-%j.out

ifconfig -a
pwd
source ../autotext/flaml_env_slurm/bin/activate
which python
python tune_vaccine.py --study-name seasonal_no_es --n-trials 200
