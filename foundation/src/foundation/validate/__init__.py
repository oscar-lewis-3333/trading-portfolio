
#chronological validation with explicit gaps and trial accounting

from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from foundation.store import EvidenceLog, platform_home, fingerprint


def _values(x, minimum=2):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError('Need a finite one-dimensional series with sufficient observations')
    
    return a


def _time(values):
    t = pd.DatetimeIndex(values)
    if t.tz is None or t.hasnans:
        raise ValueError('Timestamps must be timezone aware and non-null')
    
    return t.tz_convert('UTC')


def sharpe(values, periods_per_year=252):
    a = _values(values)
    if periods_per_year <= 0 or a.std(ddof=1) == 0:
        raise ValueError('Positive annualisation and variance required')
    
    return float(a.mean() / a.std(ddof=1) * np.sqrt(periods_per_year))


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    cutoff: pd.Timestamp


def walk_forward(times, label_end, *, min_train, test_size, purge='0D', embargo='0D', train_window=None):

    #group simultaenous decisions. gap before each test applies purge + embarge

    t, end = _time(times), _time(label_end)
    if len(t) != len(end) or not t.is_monotonic_increasing or (end < t).any():
        raise ValueError('Sorted decisions and label ends at/after decisions required')
    if min_train < 2 or test_size < 1 or (train_window is not None and train_window < min_train):
        raise ValueError('Invalid fold dimensions')
    
    gap = pd.Timedelta(purge) + pd.Timedelta(embargo)
    if pd.Timedelta(purge) < pd.Timedelta(0) or pd.Timedelta(embargo) < pd.Timedelta(0):
        raise ValueError('Negative gap')
    
    unique = t.unique()
    for start in range(min_train, len(unique), test_size):
        cutoff = unique[start] - gap
        train = np.flatnonzero((t < cutoff) & (end < cutoff))
        if train_window is not None and len(train):
            train_times = t[train].unique()[-train_window:]
            train = train[t[train].isin(train_times)]

        if len(t[train].unique()) < min_train:
            continue

        test = np.flatnonzero(t.isin(unique[start:start + test_size]))
        yield Fold(train, test, cutoff)


def circular_indices(n, block_size, resamples, seed=0):
    if n < 2 or not 1 <= block_size <= n or resamples < 2:
        raise ValueError('Invalid bootstrap dimensions')
    
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (resamples, math.ceil(n / block_size)))
    return ((starts[..., None] + np.arange(block_size)) % n).reshape(resamples, -1)[:, :n]


def _paired(strategy, benchmark):
    if isinstance(strategy, pd.Series) and isinstance(benchmark, pd.Series):
        if not strategy.index.equals(benchmark.index) or strategy.index.has_duplicates:
            raise ValueError('Paired inputs must have identical unique dates; missing dates cannot be dropped')
        
    a, b = _values(strategy), _values(benchmark)
    if a.shape != b.shape:
        raise ValueError('Unequal paired lengths')
    
    return a, b


def paired_bootstrap(strategy, benchmark, *, block_size=20, resamples=10000, seed=0, periods_per_year=252, level=.95):
    a, b = _paired(strategy, benchmark)
    if not .5 < level < 1 or periods_per_year <= 0:
        raise ValueError('Invalid confidence or annualisation')
    
    idx = circular_indices(len(a), block_size, resamples, seed)
    #bound temporary memory

    means, deltas = [], []
    for chunk in np.array_split(idx, max(1, math.ceil(resamples / 250))):
        aa, bb = a[chunk], b[chunk]
        sa, sb = aa.std(axis=1, ddof=1), bb.std(axis=1, ddof=1)
        means.extend((aa - bb).mean(axis=1))
        valid = (sa > 0) & (sb > 0)
        deltas.extend((aa.mean(axis=1)[valid] / sa[valid] - bb.mean(axis=1)[valid] / sb[valid]) * np.sqrt(periods_per_year))

    return {'mean_excess': float(np.mean(a-b)), 'mean_lower': float(np.quantile(means, 1-level)),
            'mean_upper': float(np.quantile(means, level)),
            'sharpe_difference': sharpe(a, periods_per_year) - sharpe(b, periods_per_year),
            'sharpe_lower': float(np.quantile(deltas, 1-level)) if deltas else None,
            'sharpe_upper': float(np.quantile(deltas, level)) if deltas else None,
            'valid_sharpe_draws': len(deltas), 'resamples': resamples,
            'block_size': block_size, 'level': level, 'bounds': 'one-sided', 'seed': seed}


def holm(pvalues, alpha=.05):
    p = _values(pvalues, 1)
    if ((p < 0) | (p > 1)).any() or not 0 < alpha < 1:
        raise ValueError('Invalid probabilities')
    
    order = np.argsort(p, kind='stable')
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected, corrected <= alpha


class TrialLog(EvidenceLog):
    def __init__(self, path=None):
        super().__init__(path or platform_home() / 'trial_log/trials.jsonl')

    def record(self, *, project, trial_id, protocol, sharpe_value, periods_per_year=252, status='evaluated', note=''):
        if not project or not trial_id or periods_per_year <= 0 or status not in {'evaluated','rejected','failed'}:
            raise ValueError('Invalid trial metadata')
        if sharpe_value is not None and not np.isfinite(sharpe_value):
            raise ValueError('Nonfinite Sharpe; use null for an unavailable failed trial')
        
        return self.append({'project': project, 'trial_id': trial_id, 'protocol': protocol,
                            'protocol_sha256': fingerprint(protocol), 'sharpe': sharpe_value,
                            'periods_per_year': periods_per_year, 'status': status, 'note': note})


def deflated_sharpe(returns, trial_log=None, *, periods_per_year=252):
    #deflated Sharpe probabilty and adjusted annual Sharpe estimate (seperate outputs)

    a = _values(returns, 30)
    records = [e['record'] for e in (trial_log or TrialLog()).read()]
    if not records or a.std(ddof=1) <= 0 or periods_per_year <= 0:
        raise ValueError('Need recorded trials, positive variance and annualisation')
    finite = np.array([r['sharpe'] / np.sqrt(r['periods_per_year']) for r in records if r['sharpe'] is not None])
    if not len(finite):
        raise ValueError('No finite trial Sharpe values')
    
    count = len(records)
    spread = finite.std(ddof=1) if len(finite) > 1 else 0.
    gamma = .5772156649015329
    expected = spread * ((1-gamma)*norm.ppf(1-1/count) + gamma*norm.ppf(1-1/(count*np.e))) if count > 1 else 0.
    observed = a.mean()/a.std(ddof=1)
    z = (a-a.mean())/a.std(ddof=0)
    variance = 1 - np.mean(z**3)*observed + (np.mean(z**4)-1)*observed**2/4

    if variance <= 0:
        raise ValueError('Invalid Sharpe sampling variance')
    
    probability = norm.cdf((observed-expected)*np.sqrt(len(a)-1)/np.sqrt(variance))
    return {'probability': float(probability), 'deflated_sharpe_estimate': float((observed-expected)*np.sqrt(periods_per_year)),
            'observed_sharpe': float(observed*np.sqrt(periods_per_year)),
            'expected_max_null_sharpe': float(expected*np.sqrt(periods_per_year)),
            'trials': count, 'trials_with_metrics': len(finite), 'historical_coverage': 'not_certified',
            'limitation': 'IID Sharpe-moment approximation; use block bootstrap alongside; cross-frequency trials are not directly comparable'}


def pbo(returns, *, blocks=8, periods_per_year=252, max_splits=20000):
    #CSCV probability of overfitting. higher ranking during OOS data is better

    a = np.asarray(returns, float)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all() or blocks < 4 or blocks % 2 or len(a) < 2*blocks:
        raise ValueError('Need finite candidate matrix and an even number of >=4 blocks')
    if math.comb(blocks, blocks//2) > max_splits:
        raise ValueError('CSCV split budget exceeded; freeze fewer blocks')
    
    chunks = np.array_split(np.arange(len(a)), blocks)
    logits = []
    for selected in combinations(range(blocks), blocks//2):
        train = a[np.concatenate([chunks[i] for i in selected])]
        test = a[np.concatenate([chunks[i] for i in range(blocks) if i not in selected])]
        def score(x):
            sd = x.std(axis=0, ddof=1)
            if (sd <= 0).any():
                raise ValueError('Zero variance CSCV candidate')
            
            return x.mean(axis=0)/sd*np.sqrt(periods_per_year)
        
        ins, out = score(train), score(test)
        winners = np.flatnonzero(np.isclose(ins, ins.max(), atol=1e-12, rtol=0))
        ranks = rankdata(out, method='average')/(a.shape[1]+1)
        logits.append(float(np.mean(np.log(ranks[winners]/(1-ranks[winners])))))
    return {'pbo': float(np.mean(np.asarray(logits) <= 0)), 'logits': logits, 'splits': len(logits), 'blocks': blocks}


def exposure_controls(base, exposure, *, repetitions=100, seed=0, window=20):

    #diagnostic nulls, change exposure in calendar years, and a past vol scale

    a, e = _paired(base, exposure)
    if (e < 0).any() or (e > 1).any() or window < 2 or repetitions < 1:
        raise ValueError('Filter exposure must lie in [0,1]')
    
    index = base.index if isinstance(base, pd.Series) else pd.RangeIndex(len(a))
    groups = index.year if isinstance(index, pd.DatetimeIndex) else np.zeros(len(a))
    rng = np.random.default_rng(seed)
    nulls = np.empty((len(a), repetitions))
    for j in range(repetitions):
        shuffled = e.copy()
        for group in np.unique(groups):
            ix = np.flatnonzero(groups == group); shuffled[ix] = rng.permutation(e[ix])

        nulls[:, j] = a*shuffled
    raw = pd.Series(a, index=index)
    filt = raw*e
    ratio = (filt.rolling(window).std()/raw.rolling(window).std()).shift(1).clip(0,1)
    return {'random_skip': pd.DataFrame(nulls,index=index), 'vol_matched': raw*ratio,
            'vol_scale': ratio, 'warmup_missing': ratio.isna()}


def year_spread(returns, periods_per_year=252):
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.has_duplicates:
        raise ValueError('Unique dated returns required')
    
    _values(returns)
    rows=[]
    for year, g in returns.groupby(returns.index.year):
        rows.append({'year': int(year), 'observations': len(g), 'return': float(np.prod(1+g)-1),
                     'sharpe': sharpe(g,periods_per_year) if len(g)>1 and g.std()>0 else None})
    return rows


def publication_split(returns, published_at):
    stamp = pd.Timestamp(published_at)
    if stamp.tzinfo is None or returns.index.tz is None:
        raise ValueError('Publication and return clocks must be timezone aware')
    
    return {'pre': returns.loc[returns.index < stamp], 'post': returns.loc[returns.index >= stamp]}


def turnover_capacity(trades, *, nav, adv, participation_limit=.01, periods_per_year=252, observations):
    #ADV, trade notional have to use GBP.

    notionals = _values(trades, 1)
    volumes = _values(adv, 1)
    if notionals.shape != volumes.shape or (volumes<=0).any() or nav<=0 or observations<=0 or not 0<participation_limit<=1:
        raise ValueError('Invalid capacity inputs')
    
    participation = np.abs(notionals)/volumes
    positive = participation>0
    capacity = float(nav*np.min(participation_limit/participation[positive])) if positive.any() else None

    return {'annual_one_way_turnover': float(np.abs(notionals).sum()/nav*periods_per_year/observations),
            'max_participation': float(participation.max()), 'capital_capacity': capacity,
            'within_limit': bool((participation<=participation_limit).all())}


def fixed_cluster_screen(streams, priorities, *, correlation_limit=.4):

    #frozen integer priority, then name within connected correlation components

    frame = pd.DataFrame(streams).sort_index(axis=1)
    if frame.shape[0]<3 or frame.empty or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Complete finite aligned development streams required')
    if set(frame.columns) != set(priorities) or not 0<correlation_limit<1:
        raise ValueError('Freeze a priority for every candidate')
    
    corr = frame.corr().abs()
    if corr.isna().any().any():
        raise ValueError('Undefined correlation')
    
    remaining=set(frame.columns)
    clusters=[]
    while remaining:
        todo=[min(remaining)]; component=set()
        while todo:
            name=todo.pop()
            if name in component: 
                continue
            component.add(name)
            todo.extend(n for n in remaining-component if corr.at[name,n]>=correlation_limit)
        remaining-=component

        ordered=sorted(component,key=lambda n:(priorities[n],n)); clusters.append(ordered)
    return {'selected': sorted(c[0] for c in clusters), 'clusters': clusters,
            'rule': 'connected components; frozen priority then name; development data only'}


def screen_candidates(streams,priorities,ledger_summaries,*,project,protocol,trial_log=None, correlation_limit=.4,periods_per_year=252,resamples=10000):
    #batch version of the four rule screen with deterministic clusters
    
    from foundation._research.screen import screen_candidate
    if protocol.get('screen_priorities')!=priorities:
        raise ValueError('Priorities must be frozen in protocol')
    if set(ledger_summaries)!=set(streams): 
        raise ValueError('Cost summaries required for every candidate')
    clusters=fixed_cluster_screen(streams,priorities,correlation_limit=correlation_limit)
    chosen=set(clusters['selected'])
    accepted={}
    results={}
    log=trial_log or TrialLog()
    for name in sorted(streams,key=lambda n:(priorities[n],n)):
        summary=ledger_summaries[name]
        result=screen_candidate(name,streams[name],accepted,ledger_summary=summary, max_abs_correlation=correlation_limit,periods_per_year=periods_per_year,
                                bootstrap_resamples=resamples,trial_log=None)
        if name not in chosen:
            result['reasons'].append('Not the frozen-priority representative of its correlation component')
        gross=summary.get('gross_active_return')
        cost=summary.get('cost_fraction_total')
        if gross is None or cost is None or not np.isfinite([gross,cost]).all() or gross<=0 or cost<0:
            result['reasons'].append('Cost share is undefined/nonpositive gross active return; gate not passed')
        result['accepted']=not result['reasons']
        if result['accepted']:
            accepted[name]=streams[name]
        results[name]=result
        log.record(project=project,trial_id=name,protocol=protocol,sharpe_value=result['information_ratio'] if np.isfinite(result['information_ratio']) else None,
                   periods_per_year=periods_per_year,status='evaluated' if result['accepted'] else 'rejected',note='; '.join(result['reasons']))
    return {'results':results,'accepted':sorted(accepted),'clusters':clusters}

__all__ = ["Fold", "TrialLog", "circular_indices", "deflated_sharpe", "exposure_controls", "fixed_cluster_screen", "holm", "paired_bootstrap", "pbo", "publication_split",
    "screen_candidates", "sharpe", "turnover_capacity", "walk_forward", "year_spread"]
