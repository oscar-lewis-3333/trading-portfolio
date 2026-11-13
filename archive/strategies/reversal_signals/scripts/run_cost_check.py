"""Local staged runner; never refits or reselects frozen strategies."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from reversal_costs import audit_bars,execution_arrays,replay,estimate_spreads
F=P/'data/cost_check_v1';R=P/'results/cost_check_v1';R.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load():
 m=json.loads((F/'manifest.json').read_text())
 for path,h in m['files'].items():assert sha(F/path)==h,path
 prep=pd.read_pickle(F/'original/data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl')
 cases=pd.read_pickle(F/'cases.pkl');return prep,cases

def audit():
 prep,cases=load(); p=prep['panel'];mask=audit_bars(p)
 assert p.attrs['quote_currency']=='GBP'
 for col,mcol in [('Open','open_gbp'),('Close','close_gbp')]:pd.testing.assert_series_equal(p[col],prep['market'][mcol],check_names=False)
 base={};proof=[];auditrows=[];taxrows=[];taxnames=[]
 u=pd.read_csv(F/'candidate_universe_v1.csv');tax=u.set_index('yf_symbol').sdrt_treatment
 assert tax.index.is_unique
 for name,case in cases.items():
  start=case['expected_daily'].index[0];sc=prep['schedule'].loc[start:];market=prep['market'].loc[start:]
  base[name]={}
  for role,key,exp in [('Strategy','decisions','expected_daily'),('Benchmark','benchmark','expected_benchmark')]:
   a=execution_arrays(market,sc,case[key],case['lengths']);result=replay(a)
   np.testing.assert_allclose(result['daily'].equity_gbp,case[exp],rtol=0,atol=1e-7)
   delta=result['daily'].equity_gbp.diff().fillna(0)
   np.testing.assert_allclose(delta,result['pnl'].sum(axis=1)+result['daily'].cash_interest_gbp,rtol=0,atol=1e-7)
   base[name][role]=result
   proof.append(dict(strategy=name,role=role,final_equity_gbp=result['daily'].equity_gbp.iloc[-1],max_daily_difference_gbp=float((result['daily'].equity_gbp-case[exp]).abs().max())))
   orders=result['orders'];idx=pd.MultiIndex.from_frame(orders[['Date','ticker']].drop_duplicates());selected=mask.reindex(idx)
   for flag in mask:
    auditrows.append(dict(strategy=name,role=role,flag=flag,share_days=int(selected[flag].sum()),total_share_days=len(selected),fraction=float(selected[flag].mean())))
   orders=orders.assign(treatment=orders.ticker.map(tax))
   assert orders.treatment.notna().all()
   review=orders[orders.treatment.eq('review_share_register_and_legal_status')]
   for (ticker,side),g in review.groupby(['ticker','side']):
    taxnames.append(dict(strategy=name,role=role,ticker=ticker,side=side,fills=len(g),notional_gbp=g.notional_gbp.sum(),first_date=g.Date.min(),last_date=g.Date.max()))
   for treatment,g in orders.groupby('treatment'):
    taxrows.append(dict(strategy=name,role=role,treatment=treatment,tickers=g.ticker.nunique(),fills=len(g),buy_notional_gbp=g.loc[g.side.eq('buy'),'notional_gbp'].sum(),baseline_sdrt_gbp=0))
 pd.to_pickle(base,F/'baseline_replays.pkl')
 pd.DataFrame(proof).to_csv(R/'penny_reconciliation.csv',index=False)
 pd.DataFrame(auditrows).to_csv(R/'traded_day_audit.csv',index=False)
 pd.DataFrame(taxrows).to_csv(R/'sdrt_audit.csv',index=False)
 pd.DataFrame(taxnames).to_csv(R/'sdrt_review_names_traded.csv',index=False)
 mask.sum().rename('share_days').to_csv(R/'panel_audit.csv')
 print(pd.DataFrame(proof).to_string(index=False));print(mask.sum().to_string());print(pd.DataFrame(taxrows).to_string(index=False))

if __name__=='__main__':
 if sys.argv[1]=='audit':audit()
 elif sys.argv[1]=='estimate':
  prep,cases=load();est,masks=estimate_spreads(prep['panel']);pd.to_pickle(est,F/'estimates_full_effective_spread_bps.pkl');print(est.describe(percentiles=[.1,.5,.9]).to_string())

def cost_model():
 import reversal_features as features
 prep,cases=load();p=prep['panel'];est=pd.read_pickle(F/'estimates_full_effective_spread_bps.pkl')
 from reversal_costs import fill_and_lag
 liq=(p.Close*p.Volume).unstack('ticker').shift(1).rolling(60,min_periods=60).median().stack(future_stack=True)
 # Reconstruct the frozen eligibility rule with the original feature function.
 sys.path.insert(0,str(P.parents[3]));sys.path.insert(0,str(P.parents[2]/'original_portfolio/risk_management/src'))
 import reversal_research as research
 spec=pd.read_pickle(F/'choices_and_spec.pkl')['spec'];eligible=research.build_features(prep['prices'],prep['schedule'],spec)[10].eligible
 filled,sources,buckets=fill_and_lag(est,liq,eligible)
 pd.to_pickle({'per_side':filled,'sources':sources,'buckets':buckets,'liquidity':liq,'eligible':eligible},F/'cost_model.pkl')
 # Raw estimator summaries, including the shares actually traded at the PREVIOUS close.
 rows=[];correlations=[];baseline=pd.read_pickle(F/'baseline_replays.pkl')
 for population in ['Eligible',*cases]:
  selected=eligible.copy() if population=='Eligible' else pd.Series(False,index=p.index)
  if population!='Eligible':
   orders=baseline[population]['Strategy']['orders'];dates=prep['schedule'].index
   prev=dates[dates.get_indexer(orders.Date)-1]
   idx=pd.MultiIndex.from_arrays([prev,orders.ticker],names=p.index.names).unique();selected.loc[idx]=True
  sub=est.loc[selected];bucket=buckets.loc[selected];years=sub.index.get_level_values('Date').year
  for (year,b),g in sub.groupby([years,bucket]):
   for method in est:
    v=g[method];rows.append(dict(population=population,year=year,bucket=b,estimator=method,share_days=len(g),valid_estimates=int(v.notna().sum()),median_full_effective_spread_bps=v.median(),p10_full_effective_spread_bps=v.quantile(.1),p90_full_effective_spread_bps=v.quantile(.9)))
  for window in [21,63]:
   corr=sub[[f'{m}{window}' for m in ['CS','AR','EDGE']]].corr(method='spearman')
   for x in corr:
    for y in corr:correlations.append(dict(population=population,window=window,first=x,second=y,spearman=corr.at[x,y]))
 pd.DataFrame(rows).to_csv(R/'spread_distributions.csv',index=False);pd.DataFrame(correlations).to_csv(R/'estimator_agreement.csv',index=False)
 # Matched date comparison of actual buys versus the eligible universe.
 comp=[]
 for name in cases:
  o=baseline[name]['Strategy']['orders'];o=o[o.side.eq('buy')];di=prep['schedule'].index.get_indexer(o.Date)-1
  idx=pd.MultiIndex.from_arrays([prep['schedule'].index[di],o.ticker],names=p.index.names)
  for method in est:
   universe=est[method].where(eligible).groupby(level='Date').median()
   own=est[method].reindex(idx).to_numpy();peer=universe.reindex(idx.get_level_values('Date')).to_numpy();ok=np.isfinite(own)&np.isfinite(peer)
   comp.append(dict(strategy=name,estimator=method,buy_orders=len(o),valid_comparisons=int(ok.sum()),selected_median_full_effective_spread_bps=float(np.nanmedian(own)),eligible_same_date_median_full_effective_spread_bps=float(np.nanmedian(peer[ok])),median_paired_difference_full_spread_bps=float(np.nanmedian(own-peer))))
 pd.DataFrame(comp).to_csv(R/'selected_vs_eligible.csv',index=False)
 definitions={'effective_spread_units':'basis points, full spread','execution_units':'half effective spread in bps per side, charged on net notional; round trip = two sides excluding buy-only tax','windows_sessions':[21,63],'minimum_valid_days':'ceil(0.8*window), plus ceil(0.8*(window-1)) adjacent valid pairs; no gap compression','CS':'published two-day overnight range translation, negative daily spread zeroed before rolling arithmetic mean; no stale-bar copying','AR':'two-day corrected sqrt(max(4*(c[t-1]-eta[t-1])*(c[t-1]-eta[t]),0)), then rolling mean; same convention as portfolio ETF catalogue','EDGE':'authors edge_rolling implementation, default absolute root; rolling moments; missing rows retained','point_in_time':'decision-close estimate and same-date peers shifted one full exchange session before execution','fallback':'positive eligible bucket median, positive eligible universe median, finally explicit 30bp per side house assumption; every use labelled','liquidity':'existing 60-session median GBP Close*Volume, shifted one session; buckets below 100k, 100k-250k, 250k-1m, above 1m, unknown','primary_display':'EDGE63 is a labelled reference, not selected for performance','SDRT':'0 and provisional 50bp on buys in review_share_register_and_legal_status; conditional AIM exemption assumed; historical liability NOT verified','open_premium':'1.5 times estimated per-side spread at 1x, identical numerically to 1.5x scenario; unvalidated sensitivity','choices':'all frozen at original 10bp per side; no refit or reselection','deferred_untested':['part (a) step 4 manual Trading 212 quotes','all part (b) real trades; open-as-mid and actual cost realism untested']}
 (R/'methodology.json').write_text(json.dumps(definitions,indent=2)+'\n')
 print('Cost model saved. No strategy filters or targets changed.')

if __name__=='__main__' and sys.argv[1]=='model':cost_model()

def rerun():
 from scipy.optimize import brentq
 from statsmodels.stats.multitest import multipletests
 from reversal_validation import infer_sweep_means,bootstrap_block_means
 prep,cases=load();model=pd.read_pickle(F/'cost_model.pkl');rates=model['per_side'];sources=model['sources'];buckets=model['buckets'].groupby(level='ticker').shift(1).fillna('unknown')
 universe=pd.read_csv(F/'candidate_universe_v1.csv').set_index('yf_symbol');spec=pd.read_pickle(F/'choices_and_spec.pkl')['spec']
 rows=[];annual=[];bucket_rows=[];fallback=[];breaks=[];returns={'fixed':{},'ml':{}};contrasts={'fixed':{},'ml':{}};scenario_defs={}
 scenarios={f'flat{x}bp':dict(flat=x,sdrt=False) for x in [0,10,20,30]}
 for method in rates:
  for mult in [.5,1.,1.5]:
   scenarios[f'{method}_{mult:g}x']=dict(method=method,multiple=mult,sdrt=True)
  scenarios[f'{method}_1x_noSDRT']=dict(method=method,multiple=1.,sdrt=False)
 for name,case in cases.items():
  group='fixed' if name.startswith('Fixed_') else 'ml';label=name.removeprefix('Fixed_');start=case['expected_daily'].index[0]
  market=prep['market'].loc[start:];schedule=prep['schedule'].loc[start:];a={role:execution_arrays(market,schedule,case[key],case['lengths']) for role,key in [('Strategy','decisions'),('Benchmark','benchmark')]}
  tickers=a['Strategy']['tickers'];dates=schedule.index
  tax=universe.sdrt_treatment.reindex(tickers).eq('review_share_register_and_legal_status').to_numpy()*50.
  method_rates={m:rates[m].unstack('ticker').reindex(index=dates,columns=tickers).to_numpy() for m in rates}
  for scenario,definition in scenarios.items():
   modelrate=definition.get('flat',0) if 'flat' in definition else method_rates[definition['method']]*definition['multiple']
   results={role:replay(arr,modelrate,tax if definition['sdrt'] else 0) for role,arr in a.items()}
   scenario_defs[scenario]={**definition,'cost_units':'bps per side','SDRT_rule':'50bp buy-only on unresolved review names, provisional stress' if definition['sdrt'] else 'zero SDRT (original assumption)','benchmark':'identical per-ticker/date rates and tax rule'}
   strat=results['Strategy']['daily'];bench=results['Benchmark']['daily'];excess=(strat.equity_return-bench.equity_return).iloc[1:]
   contrast=f'{label}__{scenario}';contrasts[group][contrast]=excess
   for role,result in results.items():
    daily=result['daily'];o=result['orders'];suffix='' if role=='Strategy' else '_benchmark';returns[group][f'{label}{suffix}__{scenario}']=daily.equity_return.iloc[1:]
    lag_equity=daily.equity_gbp.shift(1)
    metrics=dict(strategy=name,role=role,scenario=scenario,final_equity_gbp=daily.equity_gbp.iloc[-1],total_return_pct=100*(daily.equity_gbp.iloc[-1]/10000-1),mean_daily_net_excess_bps=excess.mean()*10000 if role=='Strategy' else np.nan,total_spread_gbp=daily.spread_gbp.sum(),total_sdrt_gbp=daily.sdrt_gbp.sum(),total_cost_gbp=daily.cost_gbp.sum(),total_notional_gbp=daily.notional_gbp.sum(),turnover_sum=daily.notional_gbp.div(lag_equity).sum(),mean_daily_turnover=daily.notional_gbp.div(lag_equity).iloc[1:].mean(),max_drawdown_pct=100*(daily.equity_gbp/daily.equity_gbp.cummax()-1).min())
    rows.append(metrics)
    np.testing.assert_allclose(daily.cost_gbp,daily.spread_gbp+daily.sdrt_gbp,rtol=0,atol=1e-8)
    np.testing.assert_allclose(daily.equity_gbp.diff().fillna(0),result['pnl'].sum(axis=1)+daily.cash_interest_gbp,rtol=0,atol=1e-7)
    for year,g in daily.iloc[1:].groupby(daily.index[1:].year):
     annual.append(dict(strategy=name,role=role,scenario=scenario,year=year,net_return_pct=100*((1+g.equity_return).prod()-1),mean_daily_net_excess_bps=excess.loc[g.index].mean()*10000 if role=='Strategy' else np.nan,spread_gbp=g.spread_gbp.sum(),sdrt_gbp=g.sdrt_gbp.sum(),notional_gbp=g.notional_gbp.sum(),turnover_sum=g.notional_gbp.div(lag_equity.loc[g.index]).sum()))
    oi=pd.MultiIndex.from_frame(o[['Date','ticker']]);o=o.assign(bucket=buckets.reindex(oi).to_numpy())
    # Additive bucket contribution (not a separately backtested sleeve).
    pnl=result['pnl'];bm=buckets.unstack('ticker').reindex(index=dates,columns=tickers)
    for bucket,g in o.groupby('bucket'):
     part=pnl.where(bm.eq(bucket),0).sum(axis=1)
     bucket_rows.append(dict(strategy=name,role=role,scenario=scenario,bucket=bucket,fills=len(g),notional_gbp=g.notional_gbp.sum(),spread_gbp=g.spread_gbp.sum(),sdrt_gbp=g.sdrt_gbp.sum(),net_stock_pnl_gbp=part.sum(),sum_daily_return_contribution_bps=10000*part.div(lag_equity).sum()))
    if 'method' in definition:
     src=sources[definition['method']].reindex(oi).to_numpy();o=o.assign(estimate_source=src)
     for source,g in o.groupby('estimate_source'):
      fallback.append(dict(strategy=name,role=role,scenario=scenario,source=source,fills=len(g),notional_gbp=g.notional_gbp.sum(),notional_fraction=g.notional_gbp.sum()/o.notional_gbp.sum()))
  # Break-even root of mean daily net matched-benchmark excess; tax fixed.
  for method,base in method_rates.items():
   def objective(x):
    st=replay(a['Strategy'],base*x,tax,record=False)['daily'].equity_return
    be=replay(a['Benchmark'],base*x,tax,record=False)['daily'].equity_return
    return float((st-be).iloc[1:].mean()*10000)
   f0=objective(0);f1=objective(1);upper=1.;fu=f1
   while fu>0 and upper<4 and np.max(base)*(upper*2)+50<10000:
    upper*=2;fu=objective(upper)
   root=brentq(objective,0,upper,xtol=1e-5) if f0>0 and fu<=0 else np.nan
   breaks.append(dict(strategy=name,estimator=method,break_even_multiple=root,zero_multiple_excess_bps=f0,one_multiple_excess_bps=f1,root_excess_bps=objective(root) if np.isfinite(root) else np.nan,bracket_upper=upper,status='root_found' if np.isfinite(root) else 'not_bracketed'))
  print('Completed',name,flush=True)
 summaries=pd.DataFrame(rows);summaries.to_csv(R/'scenario_summary.csv',index=False)
 pd.DataFrame(annual).to_csv(R/'annual_results.csv',index=False);pd.DataFrame(bucket_rows).to_csv(R/'liquidity_attribution.csv',index=False);pd.DataFrame(fallback).to_csv(R/'fallback_usage.csv',index=False);pd.DataFrame(breaks).to_csv(R/'break_even_multiples.csv',index=False)
 inference=[]
 for group,data in contrasts.items():
  data=pd.DataFrame(data);grid=pd.DataFrame(index=data.columns);inf=infer_sweep_means(data,grid,spec['primary_inference']);boot=bootstrap_block_means(data,block_length=20,n_bootstrap=5000,seed=42)*10000
  inf['bootstrap_lower_bps']=boot.quantile(.025);inf['bootstrap_upper_bps']=boot.quantile(.975)
  centered=boot-data.mean()*10000;pv=((centered.ge(data.mean()*10000).sum()+1)/5001)
  inf['bootstrap_p_one_sided']=pv;inf['bootstrap_p_holm']=multipletests(pv,method='holm')[1]
  inf['family']=group;inference.append(inf)
  out=pd.DataFrame(returns[group]).rename_axis('Date')
  # Human-readable aliases have explicit definitions; no scenario ambiguity.
  for label in (['Original','Capped'] if group=='fixed' else ['Unfiltered','XGBoost']):
   for suffix in ['', '_benchmark']:
    for mult in ['0.5','1','1.5']:
     out[f'{label}{suffix}__est{mult}x']=out[f'{label}{suffix}__EDGE63_{mult}x']
    out[f'{label}{suffix}__openPremium']=out[f'{label}{suffix}__EDGE63_1.5x']
  for method in rates:
   for column in list(out):
    if column.endswith('__'+method+'_1.5x'):
     out[column.replace('_1.5x','_openPremium')]=out[column]
  dest=P/'results/ml_extension' if group=='ml' else R
  out.to_csv(dest/('estimated_cost_daily_returns.csv' if group=='ml' else 'fixed_estimated_cost_daily_returns.csv'))
  defs={'scenarios':scenario_defs,'aliases':{'est0.5x':'EDGE63_0.5x','est1x':'EDGE63_1x','est1.5x':'EDGE63_1.5x','openPremium':'EDGE63_1.5x'},'methodology':json.loads((R/'methodology.json').read_text()),'family':group}
  defs['aliases'].update({method+'_openPremium':method+'_1.5x' for method in rates})
  (dest/('estimated_cost_scenarios.json' if group=='ml' else 'fixed_estimated_cost_scenarios.json')).write_text(json.dumps(defs,indent=2)+'\n')
 pd.concat(inference).rename_axis('contrast').to_csv(R/'inference.csv')
 print('All replays and inference completed.',flush=True)

if __name__=='__main__' and sys.argv[1]=='rerun':rerun()
