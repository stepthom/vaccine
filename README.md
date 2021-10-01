# Early stopping experiment on vaccine dataset 
Experiment to test early stopping during hyperparameter tuning of LGBM in FLAML.  See related discussion in https://github.com/microsoft/FLAML/issues/172.

# Experiment Setup

- Data is from [DrivenData's vaccine challenge](https://www.drivendata.org/competitions/66/flu-shot-learning/).
- Data contains ~27K instances and 34 features.
- Data contains both categorical and numeric features. 
- Minimal preprocessing is applied to the data.
- I used FLAML's tune api.
- I used only LGBM.
- I tuned the same hyperparameters, using the same value ranges, as FLAML does in [model.py](https://github.com/microsoft/FLAML/blob/a99e939404caeda88f32724cc264841f2f5dcfca/flaml/model.py#L215).
- For each candidate set of hyperparameter values, I ran [stratified K-fold cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) with `n_splits` set to 15 to get an accurate estimate of performance.


# Experiments

I created two experiments that share the setup above.

- Experiment 1 _did not_ use early stopping. `n_estimators` is sampled as part of the tuning process.
- Experiment 2 _did_ use early stopping. I set `n_estimators` to the upper bound (i.e., 32768). I set `early_stopping_rounds` to 100.

For each experiment, I tuned with a time budget of roughly 8 hours (30000 seconds).


# Results

Detailed results can be found in [analyze_results.ipynb](https://github.com/stepthom/vaccine/blob/main/analyze_results.ipynb). 

Some initial takeawars are as follws. Early stopping: 
- allowed more iterations/trials to be completed in the same amount of time (799 vs 192)
- found a slightly better cross validation score (0.8605 vs 0.8596)
- found the best trial faster (after 4104 seconds compared to 6621 seconds)
