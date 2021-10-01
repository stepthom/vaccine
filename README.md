# vaccine
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

I created two experiments that share the same setup above.

- Experiment 1 _did not_ use early stopping. `n_estimators` is sampled as part of the tuning process.
- Experiment 2 _did_ use early stopping. I set `n_estimators` to the upper bound (i.e., 32768). I set `early_stopping_rounds` to 100.

For each experiment, I tuned with a time budget of roughly 8 hours (30000 seconds).


# Results

Detailed results can be found in [analyze_results.ipynb](https://github.com/stepthom/vaccine/analyze_results.ipynb). Highlights include:


- Without early stopping, 192 iterations/trials were completed. With early stopping, 799 iterations/trials were completed.
- Without early stopping, the best score was found to be 0.8596. With early stopping, the best score was found to be 0.8605.
- With early stopping, the best trial was trial ID 145, found after 6621 seconds of elapsed runing. Without early stopping, the best trial was trial 48, found after 4104 seconds of elapsed runtime.


Top 10 trials without early stopping:

|     |   trial |    score |   duration |   num_leaves |   min_child_samples |   learning_rate |   max_bin |   colsample_bytree |   reg_alpha |   reg_lambda |   n_estimators |   elapsed |   is_new_best |\n|----:|--------:|---------:|-----------:|-------------:|--------------------:|----------------:|----------:|-------------------:|------------:|-------------:|---------------:|----------:|--------------:|\n|  47 |      48 | 0.860552 |    34.5268 |           21 |                  14 |       0.0954531 |         8 |           0.343715 |     1.67465 |      5.88316 |          32768 |   4104.44 |             1 |\n| 777 |     778 | 0.860408 |    37.5232 |           17 |                  15 |       0.0746648 |         8 |           0.371595 |     2.42879 |      7.82005 |          32768 |  29095.8  |             0 |\n| 741 |     742 | 0.860401 |    33.9075 |           14 |                  15 |       0.0783664 |         8 |           0.344939 |     1.26336 |      3.77804 |          32768 |  28040.7  |             0 |\n| 645 |     646 | 0.860374 |    34.2807 |           12 |                  13 |       0.100307  |         8 |           0.344881 |     1.73831 |      6.25475 |          32768 |  24993    |             0 |\n| 341 |     342 | 0.860297 |    34.5725 |           16 |                  17 |       0.0662478 |         8 |           0.352212 |     3.01802 |      9.62131 |          32768 |  15388.4  |             0 |\n| 567 |     568 | 0.86029  |    44.0434 |           20 |                  13 |       0.0601264 |         9 |           0.328983 |     1.37535 |      5.11352 |          32768 |  22516    |             0 |\n| 770 |     771 | 0.860275 |    34.684  |           16 |                  14 |       0.0907465 |        10 |           0.35557  |     1.38689 |      8.81144 |          32768 |  28863.8  |             0 |\n| 594 |     595 | 0.86026  |    35.1804 |           18 |                  12 |       0.0891454 |        10 |           0.355004 |     1.73008 |      3.26262 |          32768 |  23390.5  |             0 |\n| 318 |     319 | 0.860242 |    36.4245 |           24 |                  14 |       0.0700263 |        11 |           0.37375  |     2.77906 |      3.05017 |          32768 |  14716.9  |             0 |\n| 530 |     531 | 0.86022  |    32.6454 |           17 |                  12 |       0.0661811 |         8 |           0.329263 |     1.22484 |      4.48197 |          32768 |  21407.8  |             0 |
