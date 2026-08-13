import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats

def generate_walk_forward_splits(n_samples, train_size, test_size, step_size=None, expanding=False, embargo=0):
    #train_size : number of observations in each training window
    #test_size : number observations in each test window
    #step_size : how far to advance between folds, we default to test size giving non-overlapping test periods
    #expanding: if True, training window grows each fold rather than sliding
    #embargo : gap between train and test to prevent lables whose window overlaps the boundary from leaking information

    #we aim to get (train_idx, test_idx) positional indices

    if step_size is None:
        step_size = test_size

    splits = []
    start = 0

    while True:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        train_start = 0 if expanding else start
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))
        start += step_size

    return splits
#we use these splits to train our walk-forward model

def walk_forward_evaluate(model_fn, X, y, fwd_return, train_size=500, test_size=63, embargo=10, expanding=False):
    #model_fn: function taking no arguments which returns an unfitted model - need to train a new model each fold
    n = len(X)
    splits = generate_walk_forward_splits(n_samples=n, train_size=train_size, test_size=test_size, embargo=embargo, expanding=expanding) #splits from previous function

    fold_results = [] 
    oof_predictions = pd.Series(np.nan, index=X.index) #initialising out of fold predictions/probabilities
    oof_probabilities = pd.Series(np.nan, index=X.index)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        valid_train = ~y_train.isna()

        if valid_train.sum() < 50:
            continue

        model = model_fn()
        model.fit(X_train[valid_train], y_train[valid_train])

        valid_test = ~y_test.isna()
        if valid_test.sum() == 0:
            continue

        proba = model.predict_proba(X_test[valid_test])[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_predictions.iloc[test_idx[valid_test.values]] = pred
        oof_probabilities.iloc[test_idx[valid_test.values]] = proba

        fold_ret = fwd_return.iloc[test_idx[valid_test.values]]

        fold_results.append({
            'fold': fold_i,
            'train_start': X.index[train_idx[0]],
            'test_start': X.index[test_idx[0]],
            'test_end': X.index[test_idx[-1]],
            'n_train': valid_train.sum(),
            'n_test': valid_test.sum(),
            'accuracy': accuracy_score(y_test[valid_test], pred),
            'roc_auc': roc_auc_score(y_test[valid_test], proba) if len(set(y_test[valid_test])) > 1 else np.nan,
            'strategy_return': (fold_ret * (pred*2-1)).mean(),
        })

    return pd.DataFrame(fold_results), oof_predictions, oof_probabilities

#we repeat the above patterns but for multiple tickers to show breadth

def generate_pooled_walk_forward_splits(dates, train_size, test_size, step_size=None, expanding=False, embargo=0):
    #need to operate on unique dates, but same pattern as for 1 ticker

    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)

    if step_size is None:
        step_size = test_size

    splits = []
    start = 0
    while True:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size

        if test_end > n:
            break
        train_start = 0 if expanding else start
        train_dates = unique_dates[train_start:train_end]
        test_dates = unique_dates[test_start:test_end]
        splits.append((train_dates, test_dates))
        start += step_size

    return splits
def walk_forward_evaluate_pooled(model_fn, pooled_data, feature_cols, train_size=250, test_size=63, embargo=10, expanding=False):
    #same as for one ticker but for the pooled tickers, notice train_dates, test_dates in trading days and contribute multiple times per row

    clean = pooled_data.dropna(subset=feature_cols + ['label'])
    dates = pd.Series(clean.index)

    splits = generate_pooled_walk_forward_splits(dates=dates, train_size=train_size, test_size=test_size, embargo=embargo, expanding=expanding)

    fold_results = []

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask = clean.index.isin(train_dates)
        test_mask = clean.index.isin(test_dates)

        X_train = clean.loc[train_mask, feature_cols]
        y_train = clean.loc[train_mask, 'label']
        X_test = clean.loc[test_mask, feature_cols]
        y_test = clean.loc[test_mask, 'label']
        fwd_ret_test = clean.loc[test_mask, 'fwd_return']

        if len(X_train) < 100 or len(X_test) == 0:
            continue
        if len(set(y_test)) < 2:
            continue

        model = model_fn()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >=0.5).astype(int)
        fold_results.append({
            'fold': fold_i,
            'train_start': train_dates[0], 'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'n_train': len(X_train), 'n_test': len(X_test),
            'accuracy': accuracy_score(y_test, pred),
            'roc_auc': roc_auc_score(y_test, proba),
            'strategy_return': (fwd_ret_test * (pred*2-1)).mean(),
        })
    return pd.DataFrame(fold_results)

#we do the same thing, but only train when the model is confident.

def walk_forward_confident_only(model_fn, pooled_df, feature_cols, train_size=250, test_size=63, embargo=10, confidence_threshold=0.6):

    clean = pooled_df.dropna(subset=feature_cols + ['label'])
    dates = pd.Series(clean.index)
    splits = generate_pooled_walk_forward_splits(dates, train_size, test_size, embargo=embargo)

    rows = []
    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask, test_mask = clean.index.isin(train_dates), clean.index.isin(test_dates)
        X_train, y_train = clean.loc[train_mask, feature_cols], clean.loc[train_mask, 'label']
        X_test, y_test = clean.loc[test_mask, feature_cols], clean.loc[test_mask, 'label']
        fwd_ret = clean.loc[test_mask, 'fwd_return']

        if len(X_train) < 100 or len(set(y_test)) < 2:
            continue

        model = model_fn().fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        confident = (proba >= confidence_threshold) | (proba <= 1 - confidence_threshold)
        if confident.sum() == 0:
            continue

        pred = (proba[confident] >= 0.5).astype(int)

        y_test_confident = y_test.values[confident]
        if len(set(y_test_confident)) > 1:
            confident_auc = roc_auc_score(y_test_confident, proba[confident])
        else:
            confident_auc = np.nan
        rows.append({
            'fold': fold_i, 'n_confident': confident.sum(), 'n_total': len(proba),
            'roc_auc_all': roc_auc_score(y_test, proba) if len(set(y_test))>1 else np.nan,
            'roc_auc_confident': confident_auc,
            'strategy_return': (fwd_ret.values[confident] * (pred*2-1)).mean(),
        })
    return pd.DataFrame(rows)

#we now do the same thing, but compare to just not making a trade, instead of the difference.

def strategy_vs_buy_hold(pooled_df, feature_cols, model_fn, train_size=250, test_size=63, embargo=10, confidence_threshold=0.6):
    
    clean = pooled_df.dropna(subset=feature_cols + ['label'])
    dates = pd.Series(clean.index)
    splits = generate_pooled_walk_forward_splits(dates, train_size, test_size, embargo=embargo)

    rows = []
    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask = clean.index.isin(train_dates)
        test_mask = clean.index.isin(test_dates)

        X_train, y_train = clean.loc[train_mask, feature_cols], clean.loc[train_mask, 'label']
        X_test = clean.loc[test_mask, feature_cols]
        fwd_ret_test = clean.loc[test_mask, 'fwd_return']

        if len(X_train) < 100 or len(fwd_ret_test) == 0:
            continue

        model = model_fn().fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        confident = (proba >= confidence_threshold) | (proba <= 1 - confidence_threshold)
        if confident.sum() == 0:
            continue

        pred = (proba[confident] >= 0.5).astype(int)
        strategy_ret = (fwd_ret_test.values[confident] * (pred*2-1)).mean()
        
        buy_hold_ret = fwd_ret_test.mean()

        rows.append({
            'fold': fold_i,
            'n_confident': confident.sum(),
            'strategy_return': strategy_ret,
            'buy_hold_return': buy_hold_ret,
            'excess_return': strategy_ret - buy_hold_ret,
        })

    return pd.DataFrame(rows)

#after the plot in part 1, we investigate the behaviour directionally

def evaluate_by_direction(model_fn, pooled_df, feature_cols, train_size=250, test_size=63, embargo=10, confidence_threshold=0.6):

    clean = pooled_df.dropna(subset = feature_cols + ['label'])
    dates = pd.Series(clean.index)
    splits = generate_pooled_walk_forward_splits(dates, train_size, test_size, embargo=embargo)

    rows = []
    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask, test_mask = clean.index.isin(train_dates), clean.index.isin(test_dates)
        X_train, y_train = clean.loc[train_mask, feature_cols], clean.loc[train_mask, 'label']
        X_test = clean.loc[test_mask, feature_cols]
        fwd_ret = clean.loc[test_mask, 'fwd_return']

        if len(X_train) < 100:
            continue
        model = model_fn().fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        buys = proba >= confidence_threshold
        sells = proba <= 1 - confidence_threshold

        rows.append({
            'fold': fold_i,
            'n_buys': buys.sum(), 'n_sells': sells.sum(),
            'buy_return': fwd_ret.values[buys].mean() if buys.sum() > 0 else np.nan,
            'sell_return': fwd_ret.values[sells].mean() if sells.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)



#part 2 

def walk_forward_ranked_portfolio(model_fn, pooled_df,feature_cols, target_col='excess_return', train_size=250, test_size=63, embargo=21, top_frac=0.2, expanding=False):
    #walk forward evaluation using a regression model on our portfolio, which chooses the top ranking stocks predicted by excess returns
    #measure vs universe average

    clean = pooled_df.dropna(subset=feature_cols + [target_col, 'fwd_return'])
    dates = pd.Series(clean.index)
    splits = generate_pooled_walk_forward_splits(dates=dates, train_size=train_size, test_size=test_size, expanding=expanding, embargo=embargo)

    rows = []

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask = clean.index.isin(train_dates)
        X_train = clean.loc[train_mask, feature_cols]
        y_train = clean.loc[train_mask, target_col]

        if len(X_train) < 100:
            continue

        model = model_fn().fit(X_train, y_train)

        for test_date in test_dates:
            day_data = clean.loc[clean.index == test_date]

            if len(day_data) < 5:
                continue

            preds = model.predict(day_data[feature_cols])
            day_data = day_data.assign(predicted=preds)
            day_data = day_data.sort_values('predicted', ascending=False)

            n_select = max(1, int(len(day_data)*top_frac))
            selected = day_data.head(n_select)
            avoided = day_data.tail(n_select)

            rows.append({
                'fold': fold_i,
                'date': test_date,
                'n_universe': len(day_data),
                'n_selected': n_select,
                'top_return': selected['fwd_return'].mean(),
                'bottom_return': avoided['fwd_return'].mean(),
                'universe_return': day_data['fwd_return'].mean(),
                'top_excess': selected['excess_return'].mean(),
            })

    return pd.DataFrame(rows)

def momentum_baseline_single(pooled_df, lookback_col='return_21d', top_frac=0.2, min_universe=5, verbose=True):

#rank stocks by past return, compare top vs bottom. Only ran over 1 column, ends up being a shorter version of function below.

    clean = pooled_df.dropna(subset=['return_21d', 'excess_return', 'fwd_return'])
    results = []
    for date in clean.index.unique():
        day = clean.loc[clean.index == date]
        if len(day) < min_universe:
            continue
        day = day.sort_values(lookback_col, ascending=False)
        n = max(1, int(len(day) * top_frac))
        results.append({
            'top': day.head(n)['fwd_return'].mean(),
            'bottom': day.tail(n)['fwd_return'].mean(),
            'universe': day['fwd_return'].mean()})

    mom = pd.DataFrame(results)

    t, p = stats.ttest_rel(mom['top'], mom['bottom'])

    if verbose:
        print(f"Momentum top: {mom['top'].mean()*100:.4f}%  " f"bottom: {mom['bottom'].mean()*100:.4f}%  " f"universe: {mom['universe'].mean()*100:.4f}%")
        print(f"Top vs bottom: t={t:.3f}, p={p:.4f}")
              
    return mom, t, p


def momentum_baseline_sweep(pooled_df, lookback_cols=None, horizon_days=None, top_fracs=None, min_universe=5):

    #we check robustness of the momentum sweep affect by sweeping momentum lookback window and portfolio selection size

    #lookback_cols: which return_(X)d features to test as ranking signal, top_fracs: fraction of universe to hold long

    if lookback_cols is None:
        lookback_cols = ['return_5d', 'return_10d', 'return_21d', 'return_63d']
    if top_fracs is None:
        top_fracs = [0.1, 0.2, 0.3, 0.5]

    rows = []
    for lookback in lookback_cols:
        clean = pooled_df.dropna(subset=[lookback, 'fwd_return'])

        for frac in top_fracs:
            day_results = []
            for date in clean.index.unique():
                day = clean.loc[date == clean.index]
                if len(day) < min_universe:
                    continue
                day = day.sort_values(lookback, ascending=False)
                n = max(1, int(len(day) * frac))
                day_results.append({
                    'top': day.head(n)['fwd_return'].mean(),
                    'bottom': day.tail(n)['fwd_return'].mean(),
                    'universe': day['fwd_return'].mean(),
                })
            df_res = pd.DataFrame(day_results)
            if len(df_res) < 30:
                continue
            t_tb, p_tb = stats.ttest_rel(df_res['top'], df_res['bottom'])
            t_tu, p_tu = stats.ttest_rel(df_res['top'], df_res['universe'])

            rows.append({
                'lookback': lookback,
                'top_frac': frac,
                'n_days': len(df_res),
                'top_return': round(df_res['top'].mean()*100, 4),
                'bottom_return': round(df_res['bottom'].mean()*100, 4),
                'universe_return': round(df_res['universe'].mean()*100, 4),
                'top_minus_bottom': round((df_res['top'].mean()-df_res['bottom'].mean())*100, 4),
                'p_top_vs_bottom': round(p_tb, 4),
                'top_minus_universe': round((df_res['top'].mean()-df_res['universe'].mean())*100, 4),
                'p_top_vs_universe': round(p_tu, 4),
            })

    return pd.DataFrame(rows)

def walk_forward_momentum_rule(pooled_df, lookback_col='return_21d', train_size=250, test_size=63, embargo=21, top_frac=0.25, min_universe=5):
    #we walk-forward using just the momentum rule. rank by past return and select top 25%. 
    #embargo/train_size kept consistent since with ML pipeline even though no ML. still need embargo to seperate periods for fwd_return

    clean = pooled_df.dropna(subset=[lookback_col, 'fwd_return'])
    dates = pd.Series(clean.index)
    splits = generate_pooled_walk_forward_splits(dates, train_size, test_size, embargo=embargo)

    fold_rows = []
    for fold_i, (train_dates, test_dates) in enumerate(splits):
        test_mask = clean.index.isin(test_dates)
        test_data = clean.loc[test_mask]

        day_results = []
        for date in test_dates:
            day = test_data.loc[test_data.index==date]
            if len(day) < min_universe:
                continue

            day = day.sort_values(lookback_col, ascending=False)
            n = max(1, int(len(day) * top_frac))
            day_results.append({
                'top': day.head(n)['fwd_return'].mean(),
                'universe': day['fwd_return'].mean(),
            })

        if len(day_results) < 10:
            continue

        df_res = pd.DataFrame(day_results)
        t, p = stats.ttest_rel(df_res['top'], df_res['universe'])

        fold_rows.append({
            'fold': fold_i,
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'n_days': len(df_res),
            'top_return': df_res['top'].mean(),
            'universe_return': df_res['universe'].mean(),
            'excess': df_res['top'].mean() - df_res['universe'].mean(),
        })

    return pd.DataFrame(fold_rows)