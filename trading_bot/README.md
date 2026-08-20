# AUTONOMOUS TRADING SYSTEM

## OVERVIEW
Given all of the groundwork completed in ml_trading_signals and risk_management, how can we apply this in practice? This project turns those findings into an autonomous trading system which runs weekly using alpaca's paper trading.

## FEATURES
- Full trade pipeline - collect data, order by momentum, apply 'optimal' risk method
- Weighting scaled to be relative to other selected tickers before applying circuit breaker
- Live paper trading with alpaca
- Integrated dry-run mechanism
- Protection from duplicate orders with a pending order guard
- Automated weekly execution via cron
- Logging in a JSONL after each run
- Visualising trading history and weekly summary reports

## KEY DESIGN DECISIONS
- Chose to make weighting relative instead of as recommended by risk_management (pre circuit breaker). This is due to the sum of weights being ~0.15-0.3, and since this is the only strategy implemented currently, it should be tested on all paper funds, not just 30%.
Why weekly cron execution despite a quarterly signal horizon — circuit breaker responsiveness vs basket rebalance frequency.
- Weekly execution chosen primarily for circuit-breaker exposure changing and different weightings, which can be important week to week. The tickers selected are fixed for 63 days, as that was what was proved to be statistically significant in ml_trading_signals.
- Plan was to increase the breadth of tickers we could trade, but when testing momentum signal on all S&P 500 tickers, results came back insignificant. 'Universe' of available tickers to trade is hence limited to the selection which yielded statistical significance with momentum strategy. For more details see ml_trading_signals README.
Why the centralised path-setup pattern exists across the repo.
- New importing script introduced due to automation system. Previous iterations (with various sys.path.append) caused importing errors when used in run_bot_scheduled due to different working directories, hence paths pointed nowhere. New script written to fit across portfolio and has been inserted at the start of each project with a note.

## RESULTS
To come given enough time to see meaningful results

## LIMITATIONS
- Bot currently only implements one strategy (momentum) on a selection of tickers. Ideally would like to implement more strategies on a wider range of assets/equities
- Only one strategy is implemented on this universe, for example there could be other, more effective methods on this universe for us to compare to the momentum method that's been implemented.
- Bot only papers trades (no real capital attached) and hence no trading costs nor latency to trades aside from alpaca's modelling.

## LIBRARIES USED
- alpaca-py — connection to alpaca trading system
- python-dotenv — credential management (put details in hidden file, and extract them where needed)
- requests — alerts for discord
- pandas, numpy — creating decision pipelines
- matplotlib — visualising trading history

## PROJECT STRUCTURE
    src/
        pipeline.py              — functions to get order decisions, full running of the bot, getting a weekly summary
        broker.py                — connecting to alpaca,
        monitoring.py             — alerting through discord
        plotting_trading_bot.py  — visualising trading history
        run_bot_scheduled.py     — file that is ran by cron every week to perform trading actions
    notebooks/
        trading_bot.ipynb — main 
    data/
        bot_log.jsonl — persistent run history (JSONL, one entry per run)
        cron.log       — raw output/error log from scheduled runs

## USAGE
```python
from pipeline import generate_trading_decisions, run_bot, weekly_summary
from broker import get_client, decisions_to_orders, check_order_status
from monitoring import send_alert
from plotting_trading_bot import plot_run_history

#get the recommended trading decisions by the system
decisions, current_exposure, current_drawdown = generate_trading_decisions(tickers)

#get decisions and corresponding orders together along with an alert. submit_orders=True makes the order happen. log to log_path
decisions, orders = run_bot(tickers, submit_orders=False)

#see order status from alpaca
client = get_client(env_path='../.env')
check_order_status(client)

#analyse past data run, and print a summary of last trades
plot_run_history('../data/bot_log.jsonl')
print(weekly_summary('../data/bot_log.jsonl'))
```

automated scheduling is handled outside the notebook, in `run_bot_scheduled.py`
triggered weekly by cron:
```
0 15 * * 1 /path/to/.venv/bin/python /path/to/trading_bot/src/run_bot_scheduled.py >> /path/to/trading_bot/data/cron.log 2>&1
```