
#run a seperate paper process

import argparse
import json
from pathlib import Path
import time
from foundation.execution import Trading212Paper, RevolutXPaper, IBKRPaper, AlpacaPaper, PaperService
from foundation.costs import CostModel
from foundation.data import asof_timestamp
from foundation.risk_monitor import Limits, DrawdownStop
from foundation.store import EvidenceLog, lock, utc_now


def main():
    parser=argparse.ArgumentParser(description='Foundation paper-only execution service')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--once',action='store_true')

    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('LIVE',False):
        raise PermissionError('This service is paper only')
    
    adapters={'trading212':Trading212Paper,'revolut_x':RevolutXPaper,'ibkr':IBKRPaper,'alpaca_paper':AlpacaPaper}
    root=Path(config['state_dir']); adapter=adapters[config['venue']]
    broker=adapter(root/'paper.sqlite3',system=config['system'],currency=config['currency'],initial_cash=config['initial_cash'])
    monitor=DrawdownStop(root/'risk.json',config['maximum_drawdown'])

    service=PaperService(broker,Limits(**config['limits']),risk_monitor=monitor)
    evidence=EvidenceLog(root/'service.jsonl'); interval=float(config.get('poll_seconds',5))
    if interval<1:
        raise ValueError('Paper poll interval must be >=1 second')
    #lifetime lock must differ from service.jsonl's lock.

    with lock(root/'service.process.lock'):
        while True:
            now=asof_timestamp(utc_now())
            try:
                snapshot=json.loads(Path(config['prices_file']).read_text())
                prices=snapshot['prices']
                age=(now-asof_timestamp(snapshot['available_at'])).total_seconds()
                if age<0 or age>config.get('max_price_age_seconds',60): 
                    raise ValueError('Price snapshot future/stale')
                
                equity=broker.cash()+sum(q*prices[a] for a,q in broker.positions().items())
                monitor.update(equity,now)
                result=service.run_once(config['targets_file'],prices,at=now,cost_model=CostModel(config['flat_bps']),instruments=config['instruments'])

                equity=broker.cash()+sum(q*prices[a] for a,q in broker.positions().items())
                monitor.update(equity,now)
                evidence.append({'kind':'paper_cycle','at':now.isoformat(),'equity':equity,'positions':result['positions']})
            except (ValueError,PermissionError,KeyError,FileNotFoundError) as exc:
                evidence.append({'kind':'blocked','at':now.isoformat(),'reason':str(exc)})
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(interval)


if __name__=='__main__': main()
