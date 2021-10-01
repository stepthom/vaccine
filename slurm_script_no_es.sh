#!/bin/bash
#SBATCH --job-name=SeasonalSearcher
#SBATCH --cpus-per-task=10
#SBATCH --mem=40gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/SeasonalSearcher-%x-%j.out

ifconfig -a
pwd
source ../autotext/flaml_env_slurm/bin/activate
which python
python tune_vaccine.py --early-stopping 0 --time-budget 30000
