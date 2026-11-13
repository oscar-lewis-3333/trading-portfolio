"""Revise reporting from existing results only. No estimators or backtests run."""
from pathlib import Path
import hashlib,json,re,html
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];F=P/'data/cost_check_v1';R=P/'results/cost_check_v1';A=F/'report_revision_20261005'
LABEL='uncalibrated upper-bound sensitivity'
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
protected=[F/'manifest.json',F/'cases.pkl',F/'choices_and_spec.pkl',F/'estimates_full_effective_spread_bps.pkl',F/'cost_model.pkl',F/'baseline_replays.pkl',R/'scenario_summary.csv',R/'liquidity_attribution.csv',R/'annual_results.csv',R/'inference.csv',R/'fixed_estimated_cost_daily_returns.csv',P/'results/ml_extension/estimated_cost_daily_returns.csv',*list((P/'src').rglob('*.py'))]
before={str(p.relative_to(P)):sha(p) for p in protected}
s=pd.read_csv(R/'scenario_summary.csv');b=pd.read_csv(R/'liquidity_attribution.csv');saved_est=pd.read_pickle(F/'estimates_full_effective_spread_bps.pkl');model=pd.read_pickle(F/'cost_model.pkl');panel=pd.read_pickle(F/'original/data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl')['panel']

# Descriptive calibration, never a new spread estimate.
p=panel.loc['2025'];est=saved_est.loc['2025'];liquidity=(p.Close*p.Volume).groupby(level='ticker').median();names=liquidity.nlargest(20).index
cal=est.groupby(level='ticker').median().reindex(names)[['CS21','AR21','EDGE21']].rename(columns=lambda c:c+'_full_effective_spread_bps')
cal.insert(0,'median_daily_traded_value_gbp',liquidity.reindex(names));cal.insert(0,'ticker',names)
counts=est.CS21.groupby(level='ticker').count();cal['valid_CS21_sessions']=counts.reindex(names).to_numpy()
anchors={'GLEN.L','WPP.L','WISE.L','ENT.L','HSX.L'}
cal['plausible_quoted_spread_low_bps']=[2. if n in anchors else np.nan for n in names]
cal['plausible_quoted_spread_high_bps']=[5. if n in anchors else np.nan for n in names]
cal['comparison_basis']=['User supplied few-bp quote check; 2–5bp illustrative, not observed here' if n in anchors else 'No ticker-specific quote range supplied; manual quote required' for n in names]
for method in ['CS21','AR21','EDGE21']:
 cal[method+'_multiple_of_5bp_quote']=cal[method+'_full_effective_spread_bps']/cal.plausible_quoted_spread_high_bps
 cal[method+'_multiple_of_2bp_quote']=cal[method+'_full_effective_spread_bps']/cal.plausible_quoted_spread_low_bps
cal.to_csv(R/'calibration_top20_2025.csv',index=False)
# Same observation support within each estimator; high/low range is not a spread.
hl=10000*(p.High-p.Low)/((p.High+p.Low)/2)
rows=[]
for bucket,index in model['buckets'].loc['2025'].groupby(model['buckets'].loc['2025']).groups.items():
 for method in ['CS21','AR21','EDGE21','CS63','AR63','EDGE63']:
  ix=index[est.loc[index,method].notna()&np.isfinite(hl.loc[index])&hl.loc[index].ge(0)]
  ev=est.loc[ix,method];rv=hl.loc[ix]
  rows.append(dict(year=2025,bucket=bucket,estimator=method,matched_share_days=len(ix),median_uncalibrated_full_spread_bps=ev.median(),median_daily_high_low_range_bps=rv.median(),ratio_of_medians=ev.median()/rv.median(),spearman_with_daily_range=ev.corr(rv,method='spearman')))
range_table=pd.DataFrame(rows);range_table.to_csv(R/'calibration_range_by_bucket_2025.csv',index=False)

# Piecewise-linear crossings from the saved flat-cost grid, not new simulations.
def crossing(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if not np.isfinite(y).all():return np.nan,'missing strategy or benchmark bucket attribution',np.nan,np.nan
 if y[0]<=0:return np.nan,'nonpositive contribution at zero cost',np.nan,np.nan
 for i in range(len(x)-1):
  if y[i]>=0 and y[i+1]<=0:
   return x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]),'linear interpolation',x[i],x[i+1]
 slope=(y[-1]-y[-2])/(x[-1]-x[-2])
 if slope<0:return x[-1]-y[-1]/slope,'linear extrapolation beyond saved grid',x[-2],x[-1]
 return np.nan,'no declining crossing supported',x[-2],x[-1]
flat=s[(s.role=='Strategy')&s.scenario.str.startswith('flat')].copy();flat['cost_per_side_bps']=flat.scenario.str.extract(r'flat(\d+)bp').astype(int)
thresholds=[]
for name,g in flat.groupby('strategy',sort=False):
 g=g.sort_values('cost_per_side_bps');root,method,lo,hi=crossing(g.cost_per_side_bps,g.mean_daily_net_excess_bps)
 thresholds.append(dict(strategy=name,break_even_cost_per_side_bps=root,method=method,segment_low_per_side_bps=lo,segment_high_per_side_bps=hi,mean_daily_excess_at_0bp=g.iloc[0].mean_daily_net_excess_bps,mean_daily_excess_at_30bp=g.iloc[-1].mean_daily_net_excess_bps))
overall=pd.DataFrame(thresholds);overall.to_csv(R/'flat_break_even_per_side.csv',index=False)
bf=b[b.scenario.str.startswith('flat')].copy();bf['cost_per_side_bps']=bf.scenario.str.extract(r'flat(\d+)bp').astype(int)
wide=bf.pivot(index=['strategy','bucket','cost_per_side_bps'],columns='role',values='sum_daily_return_contribution_bps')
wide['excess_contribution_bps']=wide.Strategy-wide.Benchmark
bucket_rows=[]
for (strategy,bucket),g in wide.groupby(level=['strategy','bucket']):
 x=g.index.get_level_values('cost_per_side_bps');root,method,lo,hi=crossing(x,g.excess_contribution_bps)
 bucket_rows.append(dict(strategy=strategy,bucket=bucket,attribution_break_even_cost_per_side_bps=root,method=method,segment_low_per_side_bps=lo,segment_high_per_side_bps=hi,excess_contribution_at_0bp_bps=g.excess_contribution_bps.iloc[0],excess_contribution_at_30bp_bps=g.excess_contribution_bps.iloc[-1]))
bucket_thresholds=pd.DataFrame(bucket_rows);bucket_thresholds.to_csv(R/'flat_break_even_by_liquidity_bucket.csv',index=False)

# Relabel metadata without altering a saved numerical scenario or legacy alias.
meta_paths=[R/'methodology.json',R/'fixed_estimated_cost_scenarios.json',P/'results/ml_extension/estimated_cost_scenarios.json']
for path in meta_paths:
 data=json.loads(path.read_text());target=data.get('methodology',data)
 target['primary_display']='Corwin–Schultz (CS); CS21 headline, CS63 window sensitivity. AR and EDGE secondary.'
 target['estimate_based_scenario_status']=LABEL
 target['calibration_status']='Daily OHLC levels are biased upward versus the supplied liquid-name few-bp quote sanity range; not calibrated execution costs. Upper-bound is a stress label, not a proven statistical bound.'
 target['headline']='Flat-cost mean-excess break-even, in bp per side, from saved 0/10/20/30bp results; interpolation/extrapolation labelled.'
 target['plausible_quote_source']='User-supplied few-bp statement for GLEN/WPP/WISE/ENT/HSX; 2–5bp illustrative comparison only. Other ticker ranges unmeasured.'
 if 'scenarios' in data:
  for key,definition in data['scenarios'].items():
   definition['classification']='flat-cost sensitivity' if 'flat' in definition else LABEL
   definition['display_label']=key if 'flat' in definition else f'{LABEL}: {key}'
  data['alias_notice']='Legacy est/openPremium aliases retain their original EDGE63 numerical mappings for reproducibility; use explicit CS21/CS63 columns for the primary estimator. All non-flat aliases are uncalibrated upper-bound sensitivities.'
 path.write_text(json.dumps(data,indent=2)+'\n')

# Update report sections, keeping the existing audit and evidence tables.
def table(frame):return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:,.2f}',na_rep='Unmeasured / N/A',classes='data')
def section(title,prose,frame=None):return '<section><h2>'+html.escape(title)+'</h2><p>'+prose+'</p>'+('' if frame is None else table(frame))+'</section>'
report=(A/'original_report.html').read_text()
def replace(title,content):
 global report
 pattern=r'<section><h2>'+re.escape(title)+r'</h2>.*?</section>'
 report,n=re.subn(pattern,lambda _:content,report,flags=re.S);assert n==1,title
report=report.replace('Frozen strategy decisions, estimated execution costs. Research result; quote and fill validation remains untested.','Headline: the flat-cost break-even is approximately 20.06bp per side for the capped fixed strategy, 32.47bp for continuous Unfiltered, and 33.56bp for XGBoost. Daily OHLC spread levels are uncalibrated; they do not establish failure at realistic costs.')
headline=section('Flat-cost break-even — the live-quote test','These are costs for <strong>one buy or one sell, in bp per side</strong>, at which mean daily net excess over the identically costed benchmark reaches zero. They are derived solely from saved flat 0/10/20/30bp runs. Interpolations are approximate; the two continuous roots extrapolate the 20–30bp slope beyond the saved grid. There is no separate SDRT in those runs. A full quoted spread must be halved for a one-side crossing comparison, with any tax or other execution costs accounted for separately. A single quote is not a portfolio-wide execution test.',overall)
headline+=section('Break-even by liquidity bucket','These crossings use the saved <strong>strategy-minus-benchmark additive stock return contributions</strong> under a uniform flat cost applied to the entire portfolio. They are attribution diagnostics, not independently simulated bucket portfolios or bucket-specific-only cost stresses. They exclude cash-interest attribution and need not average to the overall threshold. Below-£100k observations arise from held shares moving across liquidity buckets. A bucket that is already nonpositive at zero cost has no positive break-even supported by the saved data. Large extrapolations are weak diagnostics, not precise trading limits.',bucket_thresholds)
report=report.replace("<section><h2>Scope and status",headline+"<section><h2>Scope and status",1)
replace('Scope and status',section('Scope and status','Reporting-only revision: no estimator, strategy, inference or backtest was re-run. The existing penny-match proof and frozen 10bp choices are retained. No strategy rules or filters changed. Corwin–Schultz is restored as primary (CS21 displayed; CS63 sensitivity). No Git commit or push was made.'))
selected=cal[['ticker','median_daily_traded_value_gbp','valid_CS21_sessions','CS21_full_effective_spread_bps','AR21_full_effective_spread_bps','EDGE21_full_effective_spread_bps','plausible_quoted_spread_low_bps','plausible_quoted_spread_high_bps','comparison_basis']]
calibration=section('Calibration sanity check: 20 most heavily traded names in 2025','Names are ranked by median daily GBP Close × Volume in the frozen 2025 panel, not by the strategy’s order sizes. The five supplied liquid-name examples have 47.8–94.8bp median full effective-spread estimates versus an illustrative 2–5bp full quoted-spread range: about <strong>10–47× higher</strong>, or roughly 43–93bp too high in full-spread units. The levels are biased upward against this sanity benchmark and cannot be used as calibrated transaction costs. Quoted and effective spreads are distinct measures; the comparison is a scale check, not a measured bias correction. The pattern is consistent with volatility contaminating the proxies, but does not isolate its cause. No common correction factor is justified for the small-cap universe. The 2–5bp range translates your “few bp” statement and is not an observed quote sample. For the other 15 names no ticker-specific quote range has been supplied: those comparisons are explicitly unmeasured. MICC has no valid spread estimate and is retained in the top-20 table rather than silently removed.',selected)
calibration+=section('Estimator median versus daily high/low range by liquidity bucket','2025, same matched share-days within each estimator. Range = 10,000 × (High − Low) / ((High + Low) / 2), in bp; this is an intraday price range, not a spread. The bucket is the saved lagged 60-session median traded-value bucket. Relative ranking information may remain useful, but neither the range association nor correlation calibrates spread levels.',range_table)
report=report.replace('<section><h2>Estimator definitions',calibration+'<section><h2>Estimator definitions',1)
replace('Units',section('Units','Outputs have nominal <strong>full effective-spread bp</strong> units but their levels are uncalibrated. Existing sensitivity replays charged half these values in <strong>bp per side</strong>; every such run is an <strong>uncalibrated upper-bound sensitivity</strong>. “Upper-bound” is a conservative stress label, not a proven bound on actual execution costs. Quoted spread is (ask − bid) / mid. A full quoted spread of 2–5bp corresponds to a 1–2.5bp half-spread per side before other costs. SDRT stress remains separate and buy-only.'))
primary=s[(s.role=='Strategy')&s.scenario.str.startswith('CS')].copy();primary.insert(1,'classification',LABEL)
replace('Main results',section('Uncalibrated upper-bound sensitivities — CS primary','The numerical results are retained for sensitivity analysis only. They do not show that reversal fails at realistic costs. CS21 is primary, CS63 is the window sensitivity; AR and EDGE remain secondary comparisons in the saved tables. Benchmarks paid the same per-share sensitivity and tax rule. The provisional SDRT stress is not verified liability. Legacy export keys and filenames remain unchanged; their JSON definitions now label every non-flat scenario and alias as an uncalibrated upper-bound sensitivity.',primary[['strategy','classification','scenario','final_equity_gbp','mean_daily_net_excess_bps','total_spread_gbp','total_sdrt_gbp']]))
report=re.sub(r'<img alt="ML equity paths.*?src="estimated_cost_equity.png">','',report)
inf=pd.read_csv(R/'inference.csv');inf=inf[inf.contrast.str.endswith('__CS21_1x')].copy();inf.insert(1,'classification',LABEL)
replace('Inference',section('Inference for uncalibrated upper-bound sensitivities','Existing saved HAC and bootstrap results only; no inference was rerun. These tests describe the chosen uncalibrated stresses, not the realism of their cost levels. CS21 is displayed. Existing settings remain Bartlett HAC with 20 lags and Holm by period family; circular bootstrap used 20-session blocks, 5,000 draws and seed 42.',inf))
replace('Break-even spread multiples',section('Secondary: multiples of uncalibrated proxies','Saved estimator-multiple roots are retained as uncalibrated upper-bound sensitivity diagnostics. They are not calibrated execution-cost thresholds. The headline uses flat-cost bp per side instead.',pd.read_csv(R/'break_even_multiples.csv').query("estimator.str.startswith('CS')",engine='python')))
rank=pd.read_csv(R/'selected_vs_eligible.csv');rank=rank[rank.estimator.str.startswith('CS')]
replace('Are actual reversal purchases wider?',section('Ranking evidence: selected shares versus eligible universe','Preserve the relative-liquidity evidence: selected reversal purchases have higher CS proxy spreads than the same-date eligible-universe median, consistent with lower liquidity on this proxy ranking. The saved comparison is against the eligible <strong>median</strong>, not an arithmetic average. Coverage and paired differences are shown. This does not validate the absolute spread levels or prove actual quoted spreads are wider.',rank))
replace('Fallback dependence',section('Fallback dependence in the uncalibrated upper-bound sensitivity','CS21 notional-weighted coverage from the existing run. Fallback values are uncalibrated peer proxies, not measured execution costs.',pd.read_csv(R/'fallback_usage.csv').query("scenario=='CS21_1x'")))
replace('Interpretation',section('Interpretation','The prior failure headline is withdrawn. The daily-data proxy levels fail the supplied liquid-name sanity check and are biased upward, so the sensitivity losses do not establish that reversal fails at realistic costs. The actionable empirical target is the flat-cost break-even in bp per side, tested against actual quotes and fills with realistic notional weighting and tax treatment. CS ranking evidence remains descriptive. Manual Trading 212 quotes (part (a), step 4) and all real trades (part (b)) remain deferred and untested, including the open-as-mid assumption.'))
# Ensure old generic prose also carries the scenario classification.
report=report.replace('Estimated-cost scenarios with SDRT','Uncalibrated upper-bound sensitivity scenarios with SDRT')
report=report.replace('every scenario by year and liquidity bucket','every flat-cost scenario and uncalibrated upper-bound sensitivity by year and liquidity bucket')
links=''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['calibration_top20_2025.csv','calibration_range_by_bucket_2025.csv','flat_break_even_per_side.csv','flat_break_even_by_liquidity_bucket.csv'])
report=report.replace('<h2>Saved evidence</h2><ul>','<h2>Saved evidence</h2><ul>'+links)
(R/'report.html').write_text(report)

# Append a bounded README draft; preserve Oscar's prior prose byte-for-byte.
def md(frame):
 def fmt(value):
  if pd.isna(value):return 'Unmeasured / N/A'
  if isinstance(value,(float,np.floating)):return f'{value:,.2f}'
  return str(value).replace('|','/').replace('\n',' ')
 lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
 lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(lines)
readme=(A/'original_README.md').read_text()
draft='''\n\n## Cost reality check — draft tables\n\nThe daily OHLC spread levels are uncalibrated and biased upward against the supplied liquid-name sanity check. The previous sensitivity losses do **not** show that reversal fails at realistic costs. All non-flat scenarios are **uncalibrated upper-bound sensitivities**; this is a stress label, not a proven upper bound on actual execution costs. Corwin–Schultz is the primary estimator (CS21 displayed; CS63 as a window sensitivity), with AR and EDGE as secondary comparisons.\n\n### Flat-cost break-even\n\nThe headline is the flat **cost per side in bp** where mean daily net excess over the equally costed benchmark reaches zero. These are approximate crossings calculated from the existing 0/10/20/30bp runs, with choices frozen at 10bp; no runs were repeated. The continuous thresholds extrapolate beyond 30bp. These runs have no separate SDRT. Live full quoted spreads must be halved before comparing a single crossing cost, with tax and other costs accounted for separately.\n\n'''
draft+=md(overall[['strategy','break_even_cost_per_side_bps','method']])+'\n\n### Liquidity-bucket attribution thresholds\n\nThese are crossings of saved strategy-minus-benchmark additive stock contributions under a uniform portfolio-wide flat cost. They are **not** independently simulated bucket strategies or the effect of charging just one bucket. Cash interest is not allocated to buckets. A missing crossing means the contribution is already nonpositive at zero cost or no declining crossing is supported. Extrapolated values are diagnostic only.\n\n'
draft+=md(bucket_thresholds[['strategy','bucket','attribution_break_even_cost_per_side_bps','method']])+'\n\n### Calibration sanity check\n\nThe five supplied examples have median full effective-spread proxies of 47.8–94.8bp versus a 2–5bp illustrative full quoted-spread range: roughly **10–47× higher**, or **43–93bp** higher in full-spread units. This is consistent with volatility contamination, not a measured universal calibration factor. The 2–5bp comparator translates the supplied “few bp” assumption; it was not observed here. Other ticker-specific plausible ranges remain unmeasured.\n\n'
draft+=md(cal[['ticker','CS21_full_effective_spread_bps','AR21_full_effective_spread_bps','EDGE21_full_effective_spread_bps','plausible_quoted_spread_low_bps','plausible_quoted_spread_high_bps']])+'\n\nNames above are the 20 largest median daily traded-value proxies in 2025. NaN means unmeasured or no valid estimate (MICC); it never means zero.\n\n'
draft+='### Proxy levels versus daily ranges\n\n2025 medians on matched observations; the daily high/low range is a price range in bp, not a spread. Full diagnostics for all estimators and both windows are in the report.\n\n'+md(range_table[range_table.estimator.eq('CS21')][['bucket','matched_share_days','median_uncalibrated_full_spread_bps','median_daily_high_low_range_bps','ratio_of_medians']])+'\n\n'
draft+='### Relative-liquidity ranking\n\nSelected purchases have higher CS proxies than the same-date eligible-universe median, consistent with lower relative liquidity. This ranking evidence is retained without treating the proxy levels as actual costs.\n\n'+md(rank[rank.estimator.eq('CS21')][['strategy','selected_median_full_effective_spread_bps','eligible_same_date_median_full_effective_spread_bps','median_paired_difference_full_spread_bps']])+'\n\n'
draft+='[Full report](results/cost_check_v1/report.html). **Deferred and untested:** the manual Trading 212 quote check and all real trades. Whether the open is a mid-price and whether the proxies reflect actual execution costs remain untested. The existing penny-match proof is retained. No strategy rules, filters, source code or numerical replay results changed in this reporting revision.\n'
(P/'README.md').write_text(readme.rstrip()+draft)
# Notebook labels only: keep its executable cells unchanged.
nbpath=P/'notebooks/reversal_cost_check.ipynb';nb=json.loads(nbpath.read_text())
for cell in nb['cells']:
 if cell['cell_type']=='markdown':
  text=''.join(cell['source'])
  if text.startswith('# Reversal cost reality check'):
   text='# Reversal cost reality check\n\nFlat-cost break-even in bp per side is the headline. CS is primary. Every non-flat result is an uncalibrated upper-bound sensitivity, not a calibrated execution-cost estimate. The levels fail the supplied liquid-name sanity check. Quotes and real trades remain deferred and untested. See the revised report and README draft tables.'
  elif text.startswith('## Scenarios'):text='## Flat costs and uncalibrated upper-bound sensitivities'
  elif text.startswith('## Break-even'):text='## Secondary uncalibrated proxy multiples\n\nHeadline flat-cost bp thresholds are in `results/cost_check_v1/flat_break_even_per_side.csv`.'
  cell['source']=text.splitlines(keepends=True)
 elif 'estimated_cost_equity.png' in ''.join(cell['source']):
  cell['source']=['display(pd.read_csv(RESULTS / "flat_break_even_per_side.csv"))']
nbpath.write_text(json.dumps(nb,indent=1)+'\n')
after={str(p.relative_to(P)):sha(p) for p in protected};assert before==after
prior=json.loads((F/'completion_manifest.json').read_text())['files']
for rel,digest in after.items():
 if rel in prior:assert digest==prior[rel],rel
record={'action':'reporting-only calibration revision','no_replays_or_estimators_run':True,'primary_estimator':'CS','protected_numerical_inputs_outputs_and_sources':after,'revision_files':{str(p.relative_to(P)):sha(p) for p in [R/'report.html',P/'README.md',*meta_paths,nbpath,R/'calibration_top20_2025.csv',R/'calibration_range_by_bucket_2025.csv',R/'flat_break_even_per_side.csv',R/'flat_break_even_by_liquidity_bucket.csv',Path(__file__).resolve()]},'prior_completion_manifest_preserved':True}
(A/'revision_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(overall.to_string(index=False));print(bucket_thresholds.to_string(index=False));print('Reporting updated; protected numerical results and source hashes unchanged.')
