from decimal import Decimal as D
import numpy as np
import pandas as pd
import pytest
from foundation.tax import match,dividends
from foundation.reporting import build_results,write_results,metrics


def t(id,day,side,q,amount,**kwargs):
    return dict(id=id,owner='o',capacity='personal',instrument_id='A',tax_date=day,side=side,quantity=q,amount_gbp=amount,**kwargs)


def test_same_day_then_30_day_then_pool_with_fees():
    rows=[t('b0','2020-01-01','buy',100,1000,fee_gbp=10),t('s','2020-02-01','sell',50,1000,fee_gbp=5),
          t('b1','2020-02-01','buy',10,150),t('b2','2020-02-20','buy',20,320)]
    r=match(rows,as_of='2020-04-01')
    assert [x['rule'] for x in r['matches']]==['same_day','30_day','section_104']
    assert D(r['total_gain_gbp'])==D(995)-D(150)-D(320)-D(202)
    assert D(r['pools'][0]['quantity'])==80 and D(r['pools'][0]['cost_gbp'])==808


def test_future_same_day_priority_beats_earlier_sale():
    rows=[t('b0','2020-01-01','buy',100,1000),t('s1','2020-02-01','sell',10,200),
          t('b','2020-02-10','buy',10,150),t('s2','2020-02-10','sell',10,210)]
    r=match(rows,as_of='2020-04-01')

    assert next(x for x in r['matches'] if x['disposal_ids']==['s1'])['rule']=='section_104'
    assert next(x for x in r['matches'] if x['disposal_ids']==['s2'])['rule']=='same_day'


@pytest.mark.parametrize('day,rule',[('2020-03-02','30_day'),('2020-03-03','section_104')])
def test_30_day_inclusive_boundary(day,rule):
    r=match([t('b0','2020-01-01','buy',10,100),t('s','2020-02-01','sell',10,200),t('b1',day,'buy',10,150)],as_of='2020-05-01')
    assert r['matches'][0]['rule']==rule


def test_incomplete_horizon_and_across_accounts():
    rows=[t('b0','2020-01-01','buy',10,100,account='broker1'),t('s','2020-02-01','sell',10,200,account='broker2')]
    with pytest.raises(ValueError):
        match(rows,as_of='2020-02-05')
    assert not match(rows,as_of='2020-02-05',final=False)['final']
    assert match(rows,as_of='2020-04-01')['total_gain_gbp']=='100'


def test_same_day_averages_and_dividends():
    rows=[t('b0','2020-01-01','buy',10,100),t('b1','2020-01-01','buy',10,200),t('s','2020-01-01','sell',10,250)]
    assert D(match(rows,as_of='2020-03-01')['total_gain_gbp'])==100
    assert dividends([dict(id='d',tax_date='2020-01-01',gross_gbp='10.10',withheld_gbp='1.50')])['records'][0]['net_cash_gbp']=='8.60'


def test_report_dual_benchmarks_initial_drawdown_and_immutability(tmp_path):
    ix=pd.date_range('2020-01-01',periods=80,tz='UTC'); rng=np.random.default_rng(1)
    s=pd.Series(rng.normal(.001,.01,80),index=ix); b=s*.8; hold=s*.5
    trades=pd.DataFrame(columns=['commission','spread_slippage','sdrt','fx','external','total'],dtype=float)
    r=build_results(project='<script>bad</script>',protocol={'v':1},strategy=s,natural_benchmark=b,buy_and_hold=hold,trades=trades,validation={'smoke':True},provenance={'synthetic_seed':1},limitations=['Synthetic only'],resamples=100,block_size=5)
    paths=write_results(tmp_path,r)
    assert '<script>bad</script>' not in paths['tear_sheet'].read_text()
    assert set(r['performance'])=={'strategy','natural_benchmark','buy_and_hold'}
    assert metrics([-.1,.01],252)['max_drawdown']==pytest.approx(-.1)
    r['project']='changed'
    with pytest.raises(FileExistsError):
        write_results(tmp_path,r)
