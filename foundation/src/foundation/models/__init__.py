
#chronological signal combination and meta labelling

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from foundation.data import utc
from foundation.validate import walk_forward, paired_bootstrap, sharpe, TrialLog


@dataclass(frozen=True)
class Protocol:
    min_train: int = 60
    retrain_every: int = 20
    window: str = 'expanding'
    train_window: int | None = None
    half_life: float | None = None
    purge: str = '0D'
    embargo: str = '0D'

    def __post_init__(self):
        if self.window not in {'rolling','expanding'} or self.min_train<2 or self.retrain_every<1:
            raise ValueError('Invalid training protocol')
        if self.window=='rolling' and (self.train_window is None or self.train_window<self.min_train):
            raise ValueError('Rolling window requires sufficient explicit history')
        if self.window=='expanding' and self.train_window is not None:
            raise ValueError('Expanding window cannot set train_window')
        if self.half_life is not None and self.half_life<=0: 
            raise ValueError('Positive decay half life required')


def decay_weights(times,half_life):
    #half-life in days passed

    t=pd.DatetimeIndex(utc(times))
    if half_life<=0 or not len(t): 
        raise ValueError('Positive half-life and observations required')
    
    age=(t.max()-t).total_seconds()/86400
    weights=np.exp2(-np.asarray(age)/half_life)
    return weights/weights.mean()


def rank_average(features,*,date_level=0):
    if not isinstance(features.index,(pd.DatetimeIndex,pd.MultiIndex)): 
        raise ValueError('Dated signal panel required')
    if np.isinf(features.to_numpy(float)).any():
        raise ValueError('Infinite feature')
    
    ranks=features.groupby(level=date_level).rank(method='average',pct=True)-.5
    score=ranks.mean(axis=1)
    return pd.DataFrame({'prediction':score,'missing_features':features.isna().sum(axis=1)})


def estimator(family='xgboost',*,classification=False,seed=0):
    if family=='ridge':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed) if classification else Ridge(alpha=10.))
    if family=='xgboost':
        from xgboost import XGBRegressor, XGBClassifier
        cls=XGBClassifier if classification else XGBRegressor
        return cls(n_estimators=80,max_depth=2,min_child_weight=10,learning_rate=.03,
                   subsample=1.,colsample_bytree=1.,reg_lambda=10.,random_state=seed,n_jobs=1,
                   tree_method='hist',**({'eval_metric':'logloss'} if classification else {'objective':'reg:squarederror'}))
    if family=='lightgbm':
        from lightgbm import LGBMRegressor,LGBMClassifier
        cls=LGBMClassifier if classification else LGBMRegressor
        return cls(n_estimators=80,max_depth=2,num_leaves=4,min_child_samples=20,learning_rate=.03,
                   reg_lambda=10.,random_state=seed,n_jobs=1,verbosity=-1,deterministic=True)
    if family=='catboost':
        from catboost import CatBoostRegressor,CatBoostClassifier
        cls=CatBoostClassifier if classification else CatBoostRegressor
        return cls(iterations=80,depth=2,learning_rate=.03,l2_leaf_reg=10.,random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False)
    
    raise ValueError('Unknown model family')


def _fit(model,X,y,weights=None):
    if hasattr(model,'steps'):
        kwargs={} if weights is None else {model.steps[-1][0]+'__sample_weight':weights}
    else: 
        kwargs={} if weights is None else {'sample_weight':weights}
    return model.fit(X,y,**kwargs)


def feature_audit(X,available_at,decision_at):
    if X.index.has_duplicates or X.columns.has_duplicates or X.empty or not np.isfinite(X.to_numpy(float)).all(): 
        raise ValueError('Complete finite feature matrix required; encode missingness explicitly')
    available=pd.DatetimeIndex(utc(available_at))
    decision=pd.DatetimeIndex(utc(decision_at))
    if len(X)!=len(available) or len(X)!=len(decision) or (available>decision).any(): 
        raise ValueError('Feature availability leakage')
    
    return {'rows':len(X),'features':list(X.columns),'constant_features':list(X.columns[X.nunique()==1]),'availability_check':'passed'}


def fit_predict(X,y,label_end,*,available_at,protocol=Protocol(),family='ridge'):
    #OOS regression scores for forward returns/alpha.

    t=X.index.get_level_values(0) if isinstance(X.index,pd.MultiIndex) else X.index
    audit=feature_audit(X,available_at,t)
    if not X.index.equals(y.index) or not np.isfinite(y.to_numpy(float)).all():
        raise ValueError('Aligned finite labels required')
    
    prediction=pd.Series(np.nan,index=X.index,name=family)
    fits=[]
    for fold in walk_forward(t,label_end,min_train=protocol.min_train,test_size=protocol.retrain_every,
                             train_window=protocol.train_window,purge=protocol.purge,embargo=protocol.embargo):
        model=estimator(family)
        weights=decay_weights(pd.DatetimeIndex(t)[fold.train],protocol.half_life) if protocol.half_life else None
        _fit(model,X.iloc[fold.train],y.iloc[fold.train],weights)
        prediction.iloc[fold.test]=model.predict(X.iloc[fold.test])
        fits.append({'cutoff':fold.cutoff.isoformat(),'training_rows':len(fold.train),
                     'last_label_end':pd.DatetimeIndex(label_end)[fold.train].max().isoformat(),
                     'test_rows':len(fold.test)})
        
    return {'prediction':prediction,'fits':fits,'audit':audit}


def compare_ladder(net_oos,control,*,project,protocol,trial_log=None,periods_per_year=252,block_size=20,resamples=10000):
    #promotion uses net OOS paired Sharpe lower bounds vs previous rung, null

    order=['rank_average','ridge','xgboost']
    order += [n for n in ['lightgbm','catboost'] if n in net_oos]
    if not set(order).issubset(net_oos):
        raise ValueError('All baseline rungs required')
    log=trial_log or TrialLog()
    results={}
    previous=None
    chain=True

    for name in order:
        values=net_oos[name]
        log.record(project=project,trial_id=name,protocol=protocol,sharpe_value=sharpe(values,periods_per_year),periods_per_year=periods_per_year)
        null=paired_bootstrap(values,control,block_size=block_size,resamples=resamples,periods_per_year=periods_per_year)
        comparison=None if previous is None else paired_bootstrap(values,net_oos[previous],block_size=block_size,resamples=resamples,periods_per_year=periods_per_year)
        passed=null['sharpe_lower'] is not None and null['sharpe_lower']>0 and (comparison is None or comparison['sharpe_lower'] is not None and comparison['sharpe_lower']>0)
        chain=bool(chain and passed)
        results[name]={'passes_research_gate':chain,'against_previous':comparison,'against_control':null}
        previous=name
    return results


def heldout_importance(model,X,y,*,seed=0,repeats=10):
    #caller supplies untouched OOS validation rows

    r=permutation_importance(model,X,y,n_repeats=repeats,random_state=seed,n_jobs=1)
    return pd.DataFrame({'feature':X.columns,'importance':r.importances_mean,'std':r.importances_std})


def prefix_invariance(builder,frame,cutoff):
    full=builder(frame)
    prefix=frame.loc[frame.index<=cutoff]
    partial=builder(prefix)
    pd.testing.assert_frame_equal(full.loc[partial.index,partial.columns],partial,check_exact=False,rtol=1e-10,atol=1e-10)
    return True


from .meta import triple_barrier, CalibratedMetaModel, cost_gated_size  #noqa: E402

__all__ = ["CalibratedMetaModel", "Protocol", "compare_ladder", "cost_gated_size",
    "decay_weights", "estimator", "feature_audit", "fit_predict", "heldout_importance",
    "prefix_invariance", "rank_average", "triple_barrier"]
