# MACHINE LEARNING FUNDAMENTALS

## OVERVIEW
In this project we cover the basic concepts of machine learning using a well-behaved dataset to better show concepts. We specifically discuss overfitting, cross-validation, the importance of not leaking data, important metrics to measure the quality of predictions and calibrating the machine. We intend to apply to financial data in a project which succeeds this one.

## FEATURES
- Train/test pipeline with proper scaling (fit on train, test on test)
- Polynomial complexity sweep with visualisation of bias-variance tradeoff
- K-fold cross-validation - comparison with single split evaluation
- Model family comparison: linear, Ridge, Lasso, decision tree, random forest, gradient boosting
- Used hold-out set to tune hyperparameters
- Classification metrics beyond accuracy — precision, recall, F1, ROC-AUC, analysis of importance
- Threshold sweeping and probability calibration
- Permutation feature importance (see which features are important without biasing using training data)
- Regularisation strength sweep (Ridge vs Lasso)

## KEY DESIGN DECISIONS
- To start discussing machine learning we use a dataset which is well behaved with a simple signal. This is to get intuition into tbe process of machine learning, to understand the basic applications on a 'nice' dataset, before entering the unknown when discussing financial data
- We choose to use permutation features importance rather than the model feature importance, as the latter is biased by training data. The former simply tests the machine with that feature removed, and measures the difference
- ROC_AUC is used for imbalanced problems primarily due to its threshold independence. ROC-AUC gives the probability of a randomly chosen positive being ranked above a randomly chosen negative, which is independent of the threshold. The accuracy gives the probability of predicting the correct class (i.e either up or down), so if we set 90% of our dataset to be positive, and set the model to predict 'up' every time, we will recieve ~90% accuracy even though its a coinflip. The ROC-AUC here will be 50%, giving us a way to differentiate between seemingly 'accurate' models.
- When tuning hyperparameters, we choose to keep a holdout set. This is to avoid bias/leaking, as when we use cross-validation, we are implicitly testing across all k-folds to find a maximum, so when we use a test set, we already know a maximum has not been achieved on this sub-dataset because else it would not be the test set. This can matter when performance between various parameters changes massively, which is not the case in the example given in the notebook.

## RESULTS

We give a list of tables that can also be found in the accompanying notebook

### Polynomial Regression Performance by Degree

| degree | n_features | train_r2 | test_r2 | gap |
|--------|------------|----------|---------|-----|
| 1 | 8 | 0.6093 | 5.958000e-01 | 1.360000e-02 |
| 2 | 44 | 0.6838 | 6.534000e-01 | 3.040000e-02 |
| 3 | 164 | 0.7436 | -1.371640e+01 | 1.446010e+01 |
| 4 | 494 | 0.7897 | -1.052243e+04 | 1.052322e+04 |
| 5 | 1286 | 0.8277 | -2.578118e+07 | 2.578118e+07 |

### Cross-Validation Metrics by Degree

| degree | cv_mean_r2 | cv_std | worst_fold | best_fold |
|--------|------------|--------|------------|-----------|
| 1 | 6.014000e-01 | 1.700000e-02 | 5.758000e-01 | 0.6213 |
| 2 | -7.466790e+01 | 1.506446e+02 | -3.759572e+02 | 0.6753 |
| 3 | -7.959967e+05 | 1.591366e+06 | -3.978729e+06 | 0.4147 |
| 4 | -5.573189e+11 | 1.114634e+12 | -2.786588e+12 | -149.0356 |

### Model Comparison (CV Performance)

| Model | CV mean R² | CV std | Worst fold |
|-------|------------|--------|------------|
| Linear regression | 0.6014 | 0.0170 | 0.5758 |
| Ridge (alpha=1) | 0.6014 | 0.0170 | 0.5758 |
| Lasso (alpha=0.1) | 0.4934 | 0.0104 | 0.4814 |
| Decision tree | 0.6138 | 0.0061 | 0.6051 |
| Random forest | 0.8097 | 0.0069 | 0.8042 |
| Gradient boosting | 0.7877 | 0.0110 | 0.7756 |

### Lasso Model Features and Values

| Feature | Value |
|---------|-------|
| MedInc | 0.7057 |
| HouseAge | 0.1060 |
| AveRooms | -0.0000 |
| AveBedrms | -0.0000 |
| Population | -0.0000 |
| AveOccup | -0.0000 |
| Latitude | -0.0112 |
| Longitude | -0.0000 |

### Best Model Hyperparameters & Scores

| Metric | Value |
|--------|-------|
| Best params | `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100}` |
| Best CV score | 0.8010 |
| Holdout score | 0.8049 |
| Optimism | -0.0038 |

### Class Balance

| Class | Count | Percentage |
|-------|-------|------------|
| Negative | 4481 | 89.6% |
| Positive | 519 | 10.4% |

### Classification Metrics by Model

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|----|---------|
| Always predict majority | 0.8940 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression | 0.9340 | 0.9167 | 0.4151 | 0.5714 | 0.7951 |
| Random forest | 0.9527 | 0.9889 | 0.5597 | 0.7149 | 0.9598 |

### Probability Calibration – Before Calibration

| index | predicted_prob | actual_freq | error |
|-------|----------------|-------------|-------|
| 0 | 0.0000 | 0.0058 | 0.0058 |
| 1 | 0.0100 | 0.0093 | -0.0007 |
| 2 | 0.0200 | 0.0000 | -0.0200 |
| 3 | 0.0300 | 0.0062 | -0.0237 |
| 4 | 0.0400 | 0.0000 | -0.0400 |
| 5 | 0.0500 | 0.0192 | -0.0308 |
| 6 | 0.0687 | 0.0164 | -0.0523 |
| 7 | 0.0989 | 0.0455 | -0.0535 |
| 8 | 0.1597 | 0.1020 | -0.0576 |
| 9 | 0.5878 | 0.8725 | 0.2847 |

### Probability Calibration – After Calibration

| index | predicted_prob | actual_freq | error |
|-------|----------------|-------------|-------|
| 0 | 0.0024 | 0.0000 | -0.0024 |
| 1 | 0.0058 | 0.0000 | -0.0058 |
| 2 | 0.0079 | 0.0204 | 0.0125 |
| 3 | 0.0100 | 0.0147 | 0.0047 |
| 4 | 0.0118 | 0.0106 | -0.0012 |
| 5 | 0.0167 | 0.0049 | -0.0118 |
| 6 | 0.0218 | 0.0122 | -0.0096 |
| 7 | 0.0370 | 0.0423 | 0.0053 |
| 8 | 0.1253 | 0.1333 | 0.0080 |
| 9 | 0.8180 | 0.8322 | 0.0142 |

### ROC-AUC Comparison

| Metric | Value |
|--------|-------|
| ROC-AUC before calibration | 0.9598 |
| ROC-AUC after calibration | 0.9535 |

### Feature Importance (Random Forest)

| feature | importance | std |
|---------|------------|-----|
| MedInc | 0.7890 | 0.0142 |
| Latitude | 0.4192 | 0.0111 |
| Longitude | 0.3063 | 0.0067 |
| AveOccup | 0.2021 | 0.0074 |
| HouseAge | 0.0701 | 0.0027 |
| AveRooms | 0.0244 | 0.0012 |
| AveBedrms | 0.0098 | 0.0006 |
| Population | 0.0079 | 0.0008 |

### Threshold Sweep – Performance Metrics

| threshold | n_signals | precision | recall | f1 |
|-----------|-----------|-----------|--------|----|
| 0.10 | 367 | 0.4060 | 0.9371 | 0.5665 |
| 0.15 | 213 | 0.6573 | 0.8805 | 0.7527 |
| 0.20 | 170 | 0.7882 | 0.8428 | 0.8146 |
| 0.25 | 149 | 0.8725 | 0.8176 | 0.8442 |
| 0.30 | 134 | 0.8806 | 0.7421 | 0.8055 |
| 0.35 | 119 | 0.9412 | 0.7044 | 0.8058 |
| 0.40 | 108 | 0.9352 | 0.6352 | 0.7566 |
| 0.45 | 99 | 0.9596 | 0.5975 | 0.7364 |
| 0.50 | 90 | 0.9889 | 0.5597 | 0.7149 |
| 0.55 | 83 | 0.9880 | 0.5157 | 0.6777 |
| 0.60 | 76 | 1.0000 | 0.4780 | 0.6468 |
| 0.65 | 65 | 1.0000 | 0.4088 | 0.5804 |
| 0.70 | 50 | 1.0000 | 0.3145 | 0.4785 |
| 0.75 | 38 | 1.0000 | 0.2390 | 0.3858 |
| 0.80 | 29 | 1.0000 | 0.1824 | 0.3085 |
| 0.85 | 21 | 1.0000 | 0.1321 | 0.2333 |
| 0.90 | 6 | 1.0000 | 0.0377 | 0.0727 |

### Ridge vs Lasso – R² and Feature Retention by Alpha

| alpha | ridge_r2 | lasso_r2 | lasso_features_kept |
|-------|----------|----------|---------------------|
| 0.00010 | 0.6035 | 0.6036 | 8 |
| 0.00023 | 0.6035 | 0.6036 | 8 |
| 0.00055 | 0.6035 | 0.6036 | 8 |
| 0.00127 | 0.6035 | 0.6037 | 8 |
| 0.00298 | 0.6035 | 0.6036 | 8 |
| 0.00695 | 0.6035 | 0.6025 | 7 |
| 0.01624 | 0.6035 | 0.5946 | 7 |
| 0.03793 | 0.6035 | 0.5653 | 5 |
| 0.08859 | 0.6035 | 0.4986 | 3 |
| 0.20691 | 0.6035 | 0.4412 | 1 |
| 0.48329 | 0.6035 | 0.2980 | 1 |
| 1.12884 | 0.6035 | -0.0001 | 0 |
| 2.63665 | 0.6035 | -0.0001 | 0 |
| 6.15848 | 0.6036 | -0.0001 | 0 |
| 14.38450 | 0.6036 | -0.0001 | 0 |
| 33.59818 | 0.6036 | -0.0001 | 0 |
| 78.47600 | 0.6033 | -0.0001 | 0 |
| 183.29807 | 0.6017 | -0.0001 | 0 |
| 428.13324 | 0.5957 | -0.0001 | 0 |
| 1000.00000 | 0.5791 | -0.0001 | 0 |

## KEY FINDINGS
- We immediately encountered the issue with single-split testing. When using single split testing on a degree 2 polynomial, we concluded that the model approximation was still pretty accurate, something which became increasingly false when we looked at cross-validation. From that table, we see that the cross-validation $R^2$ standard deviation was ~150, meaning there was wild variance in the quality of the model based upon which subset of the data was held back for testing. We conclude that the model does not predict order 2 polynomials well in general as it overfits, but one specific subset causes accurate predictions and that was the one we looked at in the beginning.
- When looking at different models, we tested a model which predicted 'up' everytime on a 10% threshold dataset. This caused the model to have ~0.9 accuracy, but the model is effectively a coinflip and tells us nothing useful about the data. This led us to a more important metric, ROC-AUC, the probability of a randomly chosen positive being chosen above a randomly chosen negative, which did correctly differentiate this model from the Random Forest model, as seen in the table above.
- As just stated, Random Forest was the best ranked model for this dataset by ROC-AUC, but when we look at the model pre-calibration, the weight of its predictions were pretty far away. This is important in finance specifically as we would like to invest proportional to our confidence. We see post calibration the weight of prediction effectively matched the actual results observed, whilst maintaining practically the same ROC-AUC as pre-calibration, making the model a lot more effective
- Lasso model was the worst performing model at predicting the California housing market. When further analysis was taken, we discovered that the model had effectively eliminated all features apart from median income and house age (slightly). When performing a feature importance test, at least to the Random Forest, we found the most important features were Median Income, Latitude and Longitude, with Latitude and Longitude combined being about as important as median income. This aligns with the results for Lasso, and suggests the reason Lasso was the worst performing model was because it was eliminating important features such as Latitude and Longitude.

## LIMITATIONS
- Dataset tested on is well-behaved with a signal. Financial data is not this clean/correlated when we eventually discuss it. We also only tested on one dataset, the same conclusions may not necessarily hold across other datasets even if they are similarly behaved.
- We only calibrated probability on one model (Random Forest) - no evidence it's as effective on other models

## LIBRARIES USED
- scikit-learn — models, cross-validation, metrics, calibration, permutation importance
- numpy, pandas — data handling
- matplotlib — complexity sweep visualisation

## PROJECT STRUCTURE
    src/
        models.py     — data prep, model evaluation, model comparison,
                         feature importance
        validation.py — cross-validation, hyperparameter tuning,
                         threshold sweeps, calibration, regularisation sweep
        plotting.py    — bias-variance tradeoff visualisation
    notebooks/
        ml_basics.ipynb — main walkthrough

## USAGE
```python
from models import prepare_data, evaluate_regression, compare_models
from validation import cross_validate_model, tune_and_evaluate, calibration_check

X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
model, results = evaluate_regression(LinearRegression(), X_train, X_test, y_train, y_test)

comparison = compare_models(X, y, k=5)

best_model, tuning_results = tune_and_evaluate(
    RandomForestRegressor(random_state=42), param_grid, X, y)
```