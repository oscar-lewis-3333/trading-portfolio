"""Units, published formulas, point-in-time timing and frozen penny reconciliation."""
import unittest,sys,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from reversal_costs import pair_spreads,execution_arrays,replay,estimate_spreads
from reversal_edge_reference import edge_rolling
from reversal_risk.execution import run_cash_interest_backtest

class EstimatorTests(unittest.TestCase):
 def test_hand_checkable_pair(self):
  b=pd.DataFrame({'Open':1.,'High':np.exp(.01),'Low':np.exp(-.01),'Close':np.exp([.0015,-.0015,.0015])},index=pd.bdate_range('2024-01-01',periods=3))
  r=pair_spreads(b)
  np.testing.assert_allclose(r.AR.iloc[1:]*10000,30)
  np.testing.assert_allclose(r.CS.iloc[1:],2*np.tanh(.01),rtol=1e-12)
  self.assertTrue(r.iloc[0].isna().all())
 def test_negative_pair_and_gap_not_bridged(self):
  b=pd.DataFrame(np.exp([[0,.005,-.005,.0015],[.02,.025,.015,.02]]),columns=['Open','High','Low','Close'])
  self.assertLess(pair_spreads(b).AR_signed_squared.iloc[1],0);self.assertEqual(pair_spreads(b).AR.iloc[1],0)
  b=pd.concat([b,b,b],ignore_index=True);b.iloc[2]=np.nan
  self.assertTrue(pair_spreads(b).iloc[2:4].isna().all().all())
 def test_edge_reference_and_units(self):
  spec=importlib.util.spec_from_file_location('edge_ref',P/'data/cost_check_v1/references/edge.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  b=pd.DataFrame({'Open':[100,102,101,103,104],'High':[103,104,103,105,106],'Low':[99,100,99,101,102],'Close':[102,101,102,104,103]})
  reference=mod.edge(*(b[c].to_numpy() for c in ['Open','High','Low','Close']))
  self.assertAlmostEqual(edge_rolling(b,5).iloc[-1],reference,places=12)
  self.assertAlmostEqual(edge_rolling(b*100,5).iloc[-1],reference,places=12)
  self.assertEqual(80/2,40) # effective spread 80bp -> per side 40bp
 def test_edge_hand_checkable_zero(self):
  b=pd.DataFrame({'Open':[100]*5,'High':[101]*5,'Low':[99]*5,'Close':[100]*5})
  self.assertAlmostEqual(edge_rolling(b,5).iloc[-1],0.,places=12)

class TimingTests(unittest.TestCase):
 def test_lagged_fallback_is_positive_and_causal(self):
  from reversal_costs import fill_and_lag
  dates=pd.bdate_range('2024-01-01',periods=4,name='Date');idx=pd.MultiIndex.from_product([dates,['A','B']],names=['Date','ticker'])
  est=pd.DataFrame({'AR21':[80.,np.nan,100.,np.nan,120.,np.nan,140.,np.nan]},index=idx)
  liq=pd.Series(200000.,index=idx);eligible=pd.Series(True,index=idx)
  filled,src,_=fill_and_lag(est,liq,eligible)
  self.assertEqual(filled.loc[(dates[1],'B'),'AR21'],40.)
  self.assertEqual(src.loc[(dates[1],'B'),'AR21'],'bucket_median')
  changed=est.copy();changed.loc[dates[2]:]=9999.
  newer,_,_=fill_and_lag(changed,liq,eligible)
  pd.testing.assert_frame_equal(filled.loc[:dates[2]],newer.loc[:dates[2]])
 def test_estimator_prefix_and_missing_minimum(self):
  dates=pd.bdate_range('2024-01-01',periods=80,name='Date');idx=pd.MultiIndex.from_product([dates,['A']],names=['Date','ticker'])
  p=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':np.tile([100.2,99.8],40),'Volume':1000.,'known_suspended':False,'close_suspended':False,'unverified_quote':False},index=idx)
  full,_=estimate_spreads(p);prefix,_=estimate_spreads(p.iloc[:50]);pd.testing.assert_frame_equal(full.iloc[:50],prefix)
  self.assertTrue(full.iloc[:20].isna().all().all())
  p.loc[(dates[30:50],'A'),'Volume']=0.
  missing,_=estimate_spreads(p);self.assertTrue(missing.loc[(dates[49],'A')].isna().all())

class LedgerTests(unittest.TestCase):
 def fixture(self):
  dates=pd.bdate_range('2024-01-01',periods=7,name='Date');idx=pd.MultiIndex.from_product([dates,['A','B']],names=['Date','ticker'])
  market=pd.DataFrame({'open_gbp':np.tile([10.,20.],7),'close_gbp':np.tile([11.,19.],7),'dividend_gbp':0.,'known_suspended':False,'open_reference_usable':True,'positive_reported_volume':True},index=idx)
  schedule=pd.DataFrame({'market_close':dates.tz_localize('UTC')+pd.Timedelta(hours=16)},index=dates)
  d=pd.DataFrame({'target_weight':[.6,.4,.4,.6,.2,.8],'signal_time':np.repeat(schedule.market_close.iloc[:3].to_numpy(),2)},index=idx[:6])
  lengths=pd.Series(1,index=dates[:3],dtype='int64');return market,schedule,d,lengths
 def test_original_engine_parity(self):
  market,schedule,d,l=self.fixture();a=execution_arrays(market,schedule,d,l)
  expected=run_cash_interest_backtest(d,market,schedule,holding_sessions_by_date=l,cost_per_side_bps=10,net_orders=True,cash_aer=.038)
  r=replay(a);np.testing.assert_allclose(r['daily'].equity_gbp,expected['daily'].equity_gbp,rtol=0,atol=1e-9)
 def test_asymmetric_costs_netting_and_reconciliation(self):
  m,s,d,l=self.fixture();a=execution_arrays(m,s,d,l);r=replay(a,np.tile([40,80],(7,1)),[50,0]);o=r['orders']
  np.testing.assert_allclose(o.spread_gbp,o.notional_gbp*o.ticker.map({'A':.004,'B':.008}))
  np.testing.assert_allclose(o.sdrt_gbp,np.where(o.side.eq('buy')&o.ticker.eq('A'),o.notional_gbp*.005,0))
  self.assertFalse(o.duplicated(['Date','ticker']).any())
  np.testing.assert_allclose(r['daily'].equity_gbp.diff().fillna(0),r['pnl'].sum(axis=1)+r['daily'].cash_interest_gbp,atol=1e-8)
 def test_frozen_penny_match(self):
  F=P/'data/cost_check_v1';prep=pd.read_pickle(F/'original/data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl');cases=pd.read_pickle(F/'cases.pkl')
  for name,c in cases.items():
   for key,exp in [('decisions','expected_daily'),('benchmark','expected_benchmark')]:
    with self.subTest(strategy=name,role=key):
     start=c[exp].index[0];a=execution_arrays(prep['market'].loc[start:],prep['schedule'].loc[start:],c[key],c['lengths']);r=replay(a,record=False)
     np.testing.assert_allclose(r['daily'].equity_gbp,c[exp],rtol=0,atol=1e-7)
     self.assertEqual(round(r['daily'].equity_gbp.iloc[-1],2),round(c[exp].iloc[-1],2))
if __name__=='__main__':unittest.main()
