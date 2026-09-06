import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, brier_score_loss


def validate_feature_cols(feature_cols, additional_forbidden=None): #use this function to check for leakage of columns in our methods

    forbidden = {
        'label',
        'fwd_return',
        'ranking_return',
        'excess_return',
        'mom_return_1d',
        'mom_entry_date',
        'mom_exit_date',
        'ticker',
        'mom_entry_open',
        'mom_high_1d',
        'mom_low_1d',
        'mom_exit_open',
    }

    if additional_forbidden is not None:
        forbidden.update(additional_forbidden)

    leaked = set(feature_cols).intersection(forbidden)

    if leaked:
        raise ValueError(f"Forbidden Columns have leaked into the feature columns: {sorted(leaked)}")
    

def hac_test(values, horizon): #used later in this file to do statistical tests across different t, p-values
    import statsmodels.api as sm

    if horizon < 1:
        raise ValueError("Horizon must be greater than (or equal to) 1")

    values = pd.Series(values).dropna() 
    if len(values) < 3:
        return np.nan, np.nan

    max_lags = min(horizon - 1, len(values) - 1)

    design_matrix = np.ones((len(values), 1))
    hac_result = sm.OLS(values.to_numpy(), design_matrix).fit(cov_type='HAC',cov_kwds={'maxlags': max_lags}, use_t=True)
    t, p = float(hac_result.tvalues[0]), float(hac_result.pvalues[0])
    return t, p 

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
    
    validate_feature_cols(X.columns)

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
        fold_ret_array = fold_ret.to_numpy(dtype=float)
        long_mask = pred == 1 #1: means long, 0 means no trade. this is long only

        long_only_returns = np.where(long_mask, fold_ret_array, 0.0) #return only when a long, else no trade and keep in capital
        long_signal_return = fold_ret_array[long_mask].mean() if long_mask.any() else np.nan
        benchmark_returns = fold_ret_array.mean()

        fold_results.append({
            'fold': fold_i,
            'train_start': X.index[train_idx[0]],
            'test_start': X.index[test_idx[0]],
            'test_end': X.index[test_idx[-1]],
            'n_train': valid_train.sum(),
            'n_test': valid_test.sum(),
            'accuracy': accuracy_score(y_test[valid_test], pred),
            'roc_auc': roc_auc_score(y_test[valid_test], proba) if len(set(y_test[valid_test])) > 1 else np.nan,
            'n_long': int(long_mask.sum()),
            'coverage': float(long_mask.mean()),
            'long_signal_return': long_signal_return,
            'long_only_return': long_only_returns.mean(),
            'benchmark_return': benchmark_returns,
            'conditional_selection_lift': long_signal_return - benchmark_returns,
            'long_only_excess': long_only_returns.mean() - benchmark_returns,
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
    validate_feature_cols(feature_cols)
 
    clean = pooled_data.dropna(subset=feature_cols + ['ticker','label', 'fwd_return'])
    dates = pd.Series(clean.index)

    splits = generate_pooled_walk_forward_splits(dates=dates, train_size=train_size, test_size=test_size, embargo=embargo, expanding=expanding)

    fold_results = []
    oos_results = [] #out of sample

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train_mask = clean.index.isin(train_dates)
        test_mask = clean.index.isin(test_dates)

        X_train = clean.loc[train_mask, feature_cols]
        y_train = clean.loc[train_mask, 'label']
        X_test = clean.loc[test_mask, feature_cols]
        y_test = clean.loc[test_mask, 'label']
        
        if len(X_train) < 100 or len(X_test) == 0:
            continue
        if y_train.nunique() < 2:
            continue

        model = model_fn()
        model.fit(X_train, y_train)

        prob_matrix = model.predict_proba(X_test)
        pos_class_index = np.where(model.classes_ == 1)[0][0]
        probs = prob_matrix[:, pos_class_index]
        pred_labels = (probs >= 0.5).astype(int)

        test_output = clean.loc[test_mask, ['ticker', 'label', 'fwd_return']].copy()
        test_output = test_output.rename_axis('date').reset_index()
        test_output.insert(0, 'fold', fold_i)
        test_output['probability'] = probs
        test_output['predicted_label'] = pred_labels

        oos_results.append(test_output)

        has_both_classes = y_test.nunique() > 1

        fold_results.append({
            'fold': fold_i,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'positive_rate': y_test.mean(),
            'mean_probability': probs.mean(),
            'max_probability': probs.max(),
            'accuracy': accuracy_score(y_test, pred_labels,),
            'roc_auc': roc_auc_score(y_test, probs) if has_both_classes else np.nan,
            'average_precision': average_precision_score(y_test, probs) if has_both_classes else np.nan,
            'brier_score': brier_score_loss(y_test, probs),
        })
    folds = pd.DataFrame(fold_results)

    if oos_results:
        oos = pd.concat(oos_results, ignore_index=True)
        oos = oos.sort_values(['date', 'ticker']).reset_index(drop=True)
    else:
        oos = pd.DataFrame()

    return folds, oos 

def evaluate_oos_top_frac(oos, top_frac=0.2, min_universe=5):
    if not 0 < top_frac < 1:
        raise ValueError("Top fraction must be between 0, 1")

    if not isinstance(min_universe, (int, np.integer)) or min_universe < 2:
        raise ValueError("Minimum universe must be at least 2, and an integer")

    required = {
        'fold',
        'date',
        'ticker',
        'label',
        'fwd_return',
        'probability',
    }

    missing = required.difference(oos.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    if oos.empty:
        raise ValueError("The OOS results are empty")

    if oos[list(required)].isna().any().any():
        raise ValueError("OOS results contain missing values")

    if oos.duplicated(['date', 'ticker']).any():
        raise ValueError("Duplicate date-ticker observations found")

    if oos.groupby('date')['fold'].nunique().gt(1).any():
        raise ValueError("A date appears in multiple test folds")

    clean = oos.copy() 
    numeric_cols = ['label', 'fwd_return', 'probability']
    clean[numeric_cols] = clean[numeric_cols].astype(float)

    if not np.isfinite(clean[numeric_cols].to_numpy()).all():
        raise ValueError("OOS results contain non-finite values")

    if not clean['label'].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1")

    if not clean['probability'].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1")

    rows = []

    for date, day in clean.groupby('date', sort=True):
        if len(day) < min_universe:
            continue

        day = day.sort_values(['probability', 'ticker'], ascending=[False, True])
        n = max(1, int(len(day) * top_frac))

        selected = day.head(n)

        top_return = selected['fwd_return'].mean()
        universe_return = day['fwd_return'].mean()

        top_label_rate = selected['label'].mean()
        universe_label_rate = day['label'].mean()

        rows.append({
            'fold': day['fold'].iloc[0],
            'date': date,
            'n_universe': len(day),
            'n_selected': n,
            'top_label_rate': top_label_rate,
            'universe_label_rate': universe_label_rate,
            'label_lift': top_label_rate - universe_label_rate,
            'top_return': top_return,
            'universe_return': universe_return,
            'excess_return': top_return - universe_return,
        })
    daily = pd.DataFrame(rows)

    if daily.empty:
        raise ValueError("No dates contained required minimum universe")

    return daily

def evaluate_oos_score(oos, n_buckets=5, min_universe=5):
    if not isinstance(n_buckets, (int, np.integer)) or n_buckets < 2:
        raise ValueError("Number of buckets must be greater than (or equal to) 2 and an integer")

    if not isinstance(min_universe, (int, np.integer)) or min_universe < n_buckets:
        raise ValueError("Minimum universe must be at least the number of buckets and an integer")

    required = {
        'fold',
        'date',
        'ticker',
        'label',
        'fwd_return',
        'probability',
    }

    missing = required.difference(oos.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    if oos.empty:
        raise ValueError("The OOS results are empty")
    if oos[list(required)].isna().any().any():
        raise ValueError("OOS results contain missing values")
    if oos.duplicated(['date', 'ticker']).any():
        raise ValueError("Duplicate date-ticker observations found")

    clean = oos.copy()
    numeric_cols = ['label', 'fwd_return', 'probability']
    clean[numeric_cols] = clean[numeric_cols].astype(float)

    if not np.isfinite(clean[numeric_cols].to_numpy()).all():
        raise ValueError("OOS results contain non-finite values")
    if not clean['label'].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1")
    if not clean['probability'].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1")

    rows = []

    for date, day in clean.groupby('date', sort=True):
        if len(day) < min_universe:
            continue

        if day['fold'].nunique() != 1:
            raise ValueError("A date appears in multiple folds")

        day = day.sort_values(['probability', 'ticker'], ascending=[True, True]).copy()
        day['bucket'] = (np.arange(len(day)) * n_buckets // len(day)) + 1


        for bucket, group in day.groupby('bucket'):
            rows.append({
                'fold': day['fold'].iloc[0],
                'date': date,
                'bucket': int(bucket),
                'n_stocks': len(group),
                'mean_probability': group['probability'].mean(),
                'label_rate': group['label'].mean(),
                'mean_return': group['fwd_return'].mean(),
            })
    daily_bucket = pd.DataFrame(rows)
    if daily_bucket.empty:
        raise ValueError("No dates met the universe requirement")

    bucket_summary = daily_bucket.groupby('bucket').agg(n_dates=('date', 'nunique'),
            mean_stocks=('n_stocks', 'mean'),
            mean_probability=('mean_probability', 'mean'),
            label_rate=('label_rate', 'mean'),
            mean_return=('mean_return', 'mean'),).reset_index()

    return daily_bucket, bucket_summary
        
            
    
#part 2 

def walk_forward_ranked_portfolio(model_fn, pooled_df,feature_cols, target_col='excess_return', train_size=250, test_size=63, embargo=21, top_frac=0.2, expanding=False):
    #walk forward evaluation using a regression model on our portfolio, which chooses the top ranking stocks predicted by excess returns
    #measure vs universe average

    validate_feature_cols(feature_cols, additional_forbidden={target_col})
    
    clean = pooled_df.dropna(subset=feature_cols + [target_col, 'ranking_return'])
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
                'top_return': selected['ranking_return'].mean(),
                'bottom_return': avoided['ranking_return'].mean(),
                'universe_return': day_data['ranking_return'].mean(),
                'top_excess': selected[target_col].mean(),
            })

    return pd.DataFrame(rows)

def momentum_baseline_single(pooled_df, lookback_col='return_21d', outcome_col='ranking_return', horizon=21, top_frac=0.2, min_universe=5, verbose=True):


#rank stocks by past return, compare top vs bottom. Only ran over 1 column, ends up being a shorter version of function below.

    if not 0 < top_frac <=1 or horizon < 1:
        raise ValueError("Horizon cannot be less than 1, and top_frac must be positive and less than (or equal to) 1")
    

    clean = pooled_df.dropna(subset=[lookback_col, outcome_col])
    results = []

    for date in sorted(clean.index.unique()):

        day = clean.loc[clean.index == date]

        if len(day) < min_universe:
            continue

        day = day.sort_values(lookback_col, ascending=False)
        n = max(1, int(len(day) * top_frac))
        results.append({
            'date': date,
            'top': day.head(n)[outcome_col].mean(),
            'bottom': day.tail(n)[outcome_col].mean(),
            'universe': day[outcome_col].mean()})

    mom = pd.DataFrame(results)

    if mom.empty:
        return mom, np.nan, np.nan

    mom = mom.set_index('date').sort_index()
    diff = mom['top'] - mom['bottom']

    t, p = hac_test(diff, horizon=horizon)

    if verbose:
        print(f"Momentum top: {mom['top'].mean()*100:.4f}%  " f"bottom: {mom['bottom'].mean()*100:.4f}%  " f"universe: {mom['universe'].mean()*100:.4f}%")
        print(f"Top vs bottom (HAC adjusted): t={t:.3f}, p={p:.4f}")
              
    return mom, t, p


def momentum_baseline_sweep(pooled_df, lookback_cols=None, horizon=21, top_fracs=None, outcome_col='ranking_return', min_universe=5):

    from statsmodels.stats.multitest import multipletests

    #we check robustness of the momentum sweep affect by sweeping momentum lookback window and portfolio selection size

    #lookback_cols: which return_(X)d features to test as ranking signal, top_fracs: fraction of universe to hold long

    if horizon < 1:
        raise ValueError("Horizon must be greater than (or equal to) 1")
    

    if lookback_cols is None:
        lookback_cols = ['return_5d', 'return_10d', 'return_21d', 'return_63d']

    if top_fracs is None:
        top_fracs = [0.1, 0.2, 0.3, 0.5]

    if any(frac <=0 or frac > 1 for frac in top_fracs):
        raise ValueError("All top fractions must be between 0 and 1")   

    rows = []
    for lookback in lookback_cols:
        clean = pooled_df.dropna(subset=[lookback, outcome_col])

        for frac in top_fracs:
            day_results = []
            for date in sorted(clean.index.unique()):
                day = clean.loc[date == clean.index]
                if len(day) < min_universe:
                    continue
                day = day.sort_values(lookback, ascending=False)
                n = max(1, int(len(day) * frac))
                day_results.append({
                    'top': day.head(n)[outcome_col].mean(),
                    'bottom': day.tail(n)[outcome_col].mean(),
                    'universe': day[outcome_col].mean(),
                })
            df_res = pd.DataFrame(day_results)
            if len(df_res) < 30:
                continue

            tb = df_res['top'] - df_res['bottom'] #top - bottom
            tu = df_res['top'] - df_res['universe'] #top - universe

            t_tb, p_tb = hac_test(tb, horizon=horizon)
            t_tu, p_tu = hac_test(tu, horizon=horizon)

            rows.append({
                'lookback': lookback,
                'top_frac': frac,
                'n_days': len(df_res),
                'top_return': round((df_res['top'].mean() * 100),4),
                'bottom_return': round((df_res['bottom'].mean()* 100),4),
                'universe_return': round((df_res['universe'].mean() * 100),4),
                'top_minus_bottom': round((tb.mean() * 100), 4),
                'top_minus_universe': round((tu.mean() * 100), 4),
                't_top_vs_bottom': round((t_tb), 4),
                'p_top_vs_bottom_raw':p_tb,
                't_top_vs_universe': round((t_tu), 4),
                'p_top_vs_universe_raw':p_tu,  #deliberately do not round p-values before holm-adjusting
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    
    def holm_adjust(p_values):
        p_values = pd.Series(p_values)
        adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
        valid = p_values.notna()

        if valid.any():
            adjusted.loc[valid]= multipletests(p_values.loc[valid], method='holm')[1]

        return adjusted

    
    results['p_top_vs_bottom'] = holm_adjust(results['p_top_vs_bottom_raw'])
    results['p_top_vs_universe'] = holm_adjust(results['p_top_vs_universe_raw'])
    round_columns = { 
        'p_top_vs_bottom_raw',
        'p_top_vs_bottom',
        'p_top_vs_universe_raw',
        'p_top_vs_universe'
    }
    for column in round_columns:
        results[column] = results[column].round(4)

    return results

def walk_forward_momentum_rule(pooled_df, lookback_cols=None, top_fracs=None, outcome_col='ranking_return', horizon=21, train_size=504, test_size=63, embargo=None, expanding=True, min_universe=5):

    #new momentum rule. after reducing lookahead bias, significance of topvs bottom dissappeared, but have top vs universe significance for top 10%, 20% of . on each selects tickers using  momentum rule on training data, before using same on test data.

    if horizon < 1 or train_size < 1 or test_size < 1:
        raise ValueError("Horizon, train size and test size must all be at least 1")

    if lookback_cols is None:
        lookback_cols = ['return_5d', 'return_10d', 'return_21d', 'return_63d']
    if top_fracs is None:
        top_fracs = [0.1, 0.2, 0.3, 0.5] #this gives 16 possible strategies as in the sweep. It is NOT assumed which one is the best.
    
    if not lookback_cols or  not top_fracs or any(frac <= 0 or frac > 1 for frac in top_fracs):
        raise ValueError("At least one lookback column is required, and all top fractions must be between 0 and 1")

    if embargo is None:
        embargo = horizon

    if embargo < horizon:
        raise ValueError("Embargo must be at least as large as the forward-return horizon")


    required = set(lookback_cols) | {outcome_col}
    missing = required.difference(pooled_df.columns)
    if missing:
        raise ValueError(f"Walk-forward columns are missing: {sorted(missing)}")

    #same eligible dates for each candidate - removes issue of something being favoured because of the time it was tested on
    eligible = pooled_df.dropna(subset=list(lookback_cols) + [outcome_col])
    dates = pd.Series(eligible.index)
    splits = generate_pooled_walk_forward_splits(dates=dates, train_size=train_size, test_size=test_size, expanding=expanding, embargo=embargo,) #as before, 21 day embargo used as an overlap buffer to stop leakage from training into testing

    def daily_top_vs_universe(period_dates, lookback_col, top_frac): #evaluate momentum rule over a given period (train/test) and given top fraction
        columns = [lookback_col, outcome_col]
        period = eligible.loc[eligible.index.isin(period_dates), columns].copy() #collection of eligible dates and wanted columns
        if period.empty:
            return pd.DataFrame()

        period['_date'] = period.index
        period = period.sort_values(['_date', lookback_col], ascending=[True, False]) #go forward in time, descending in lookback_col (returns over some time period)
        period['n_universe'] = period.groupby('_date')[outcome_col].transform('size') #amount of stocks available
        period = period.loc[period['n_universe'] >= min_universe]

        if period.empty:
            return pd.DataFrame()

        period['rank'] = period.groupby('_date').cumcount() #0 is best, 1 second best, etc.
        period['n_selected'] = (period['n_universe'] * top_frac).astype(int).clip(lower=1) 
        selected = period.loc[period['rank'] < period['n_selected']] 

        universe_return = period.groupby('_date')[outcome_col].mean() #universe/top returns per eligible date
        top_return = selected.groupby('_date')[outcome_col].mean()
        n_universe = period.groupby('_date')['n_universe'].first().astype(int)
        n_selected = selected.groupby('_date').size().astype(int)

        daily = pd.concat([top_return.rename('top_return'),
                universe_return.rename('universe_return'),
                n_universe.rename('n_universe'),
                n_selected.rename('n_selected'),],
            axis=1,).dropna()
        
        daily['excess'] = daily['top_return'] - daily['universe_return']
        daily.index.name = 'date'
        return daily.sort_index()

    all_dates = np.sort(eligible.index.unique())
    daily_cache = {(lookback_col, top_frac): daily_top_vs_universe(all_dates, lookback_col, top_frac,) 
                   for lookback_col in lookback_cols for top_frac in top_fracs} #collect all possible combinations and corresponding data
    
    fold_rows = []
    daily_rows = []

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        candidates = []

        for lookback_col in lookback_cols:
            for top_frac in top_fracs:
                candidate_daily = daily_cache[(lookback_col, top_frac)] #get data for the given lookback_col, top_frac
                train_daily = candidate_daily.loc[candidate_daily.index.isin(train_dates)] 
                if len(train_daily) < 30:
                    continue

                train_t, train_p = hac_test(train_daily['excess'], horizon=horizon)
                if not np.isfinite(train_t):
                    continue

                candidates.append({
                    'lookback': lookback_col,
                    'top_frac': top_frac,
                    'train_mean_excess': train_daily['excess'].mean(),
                    'train_t_hac': train_t,
                    'train_p_raw': train_p,
                })

        if not candidates:
            continue

        positive_candidates = [row for row in candidates if row['train_mean_excess'] > 0]
        selection_pool = positive_candidates if positive_candidates else candidates #try to get the positive candidates if possible
        best = max(selection_pool, key=lambda row: (row['train_t_hac'], row['train_mean_excess']),) #primary selection is the t value, then excess, favouring a choice relative to its variability 

        best_daily = daily_cache[(best['lookback'], best['top_frac'])]
        test_daily = best_daily.loc[best_daily.index.isin(test_dates)] #stick with this selection for rest of the fold
        if test_daily.empty:
            continue

        fold_rows.append({
            'fold': fold_i,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_daily.index[0],
            'test_end': test_daily.index[-1],
            'lookback': best['lookback'],
            'top_frac': best['top_frac'],
            'train_mean_excess': best['train_mean_excess'],
            'train_t_hac': best['train_t_hac'],
            'train_p_raw': best['train_p_raw'],
            'n_test_days': len(test_daily),
            'top_return': test_daily['top_return'].mean(),
            'universe_return': test_daily['universe_return'].mean(),
            'excess': test_daily['excess'].mean(),
        })

        test_output = test_daily.reset_index()
        test_output.insert(0, 'fold', fold_i)
        test_output['lookback'] = best['lookback']
        test_output['top_frac'] = best['top_frac']
        daily_rows.append(test_output)

    folds = pd.DataFrame(fold_rows)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return folds, daily
