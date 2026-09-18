import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance


def prepare_data(X, y, test_size=0.3, random_state=42, scale=True):
    #split into train/test data and optional features to standardise

    #scaler fitted on training data only, then applied to both sets, fitting on full set leaks info about test set into training set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) #compute mean, standard deviation and apply them
        X_test = scaler.transform(X_test) #apply already computed values
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler

def evaluate_regression(model, X_train, X_test, y_train, y_test, label=""):
    #fit model and report training, test performance. Gap between is called overfitting diagnostic

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    results = {
        'model': label or type(model).__name__,
        'train_r2': r2_score(y_train, pred_train),
        'test_r2': r2_score(y_test, pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, pred_test)),
    }
    results['r2_gap'] = results['train_r2'] - results['test_r2']

    return model, results
def polynomial_complexity_sweep(X, y, max_degree=5, test_size=0.3, random_state=42):
    #plot polynomial regressions and see training, test performances diverage as degree increases

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    rows = []
    for degree in range(1, max_degree+1):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())

        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        rows.append({
            'degree': degree,
            'n_features': model.named_steps['polynomialfeatures'].n_output_features_,
            'train_r2': round(train_r2, 4),
            'test_r2': round(test_r2, 4),
            'gap': round(train_r2 - test_r2, 4)
        })

    return pd.DataFrame(rows).set_index('degree')

def compare_models(X, y, k=5, random_state=42):
    #aim to cross-validate model families on same data. such families include Ridge, Lasso (regularised linear regression), two ensemble methods and a decision tree

    from validation import cross_validate_model #import function which splits data and test on all, before compiling table

    models = {
        'Linear regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge (alpha=1)':   make_pipeline(StandardScaler(), Ridge(alpha=1.0)), #regularised L.R models which penalise large coefficients.
        'Lasso (alpha=0.1)': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
        'Decision tree':     DecisionTreeRegressor(random_state=random_state), #if no depth limit will grow until all data seperated
        'Random forest':     RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1), #train many trees on random subset of rows/features, averaging removes outliers. n_jobs=-1 uses all CPU cores since 100 trees too parallel
        'Gradient boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state), #trains trees sequentially on the errors of the previous tree - can be prone to overfitting if not tuned
    }
    rows = []
    for name, model in models.items():
        cv = cross_validate_model(model, X, y, k=k)
        rows.append({
            'Model': name,
            'CV mean R²': round(cv['mean'], 4),
            'CV std': round(cv['std'], 4),
            'Worst fold': round(cv['scores'].min(), 4),
        })

    return pd.DataFrame(rows).set_index('Model') #models as row names

#so far we have only looked at regression, now we look at classification, and the corresponding metrics

def evaluate_classifier(model, X_train, X_test, y_train, y_test, label=""):
    #we fit a classifier, and report metrics. accuracy alone can be misleading - a model predicting the majority each time can score highly while producing nonsense

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] #predicting probability of 'positive class'

    return model, {
        'model': label or type(model).__name__,
        'accuracy': round(accuracy_score(y_test, pred), 4), #fraction correct, fails badly as discussed above
        'precision': round(precision_score(y_test, pred, zero_division=0), 4), #of cases predicted positive, how many were actually positive
        'recall': round(recall_score(y_test, pred, zero_division=0), 4), #of actual positives, how many did we predict?
        'f1': round(f1_score(y_test, pred, zero_division=0), 4), #harmonic mean of precision, recall
        'roc_auc': round(roc_auc_score(y_test, proba), 4), #probability that a random chosen positive is ranked above a randomly chosen negative. Threshold independent, that is it measures whether the model ranks correctly regardless of where you set the buy/sell cutoff
    }
def feature_importance(model, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    #shuffle features to see how bad performance drops when removed. preferred over a model's build in importances which are biased towards high-cardinality features

    result = permutation_importance(model, X_test, y_test,n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    return pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean.round(4),
        'std': result.importances_std.round(4),
    }).sort_values('importance', ascending=False).set_index('feature')

