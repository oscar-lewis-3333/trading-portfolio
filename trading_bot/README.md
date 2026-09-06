# AUTONOMOUS TRADING SYSTEM

## OVERVIEW
Given all of the groundwork completed in ml_trading_signals and risk_management, how can we apply this in practice? This project turns those findings into an autonomous trading system which runs daily using alpaca's paper trading.

## FEATURES
- Full trade pipeline - collect data, order by 63d momentum, rebalances handled correctly
- Weighting scaled to be relative to other selected tickers before applying circuit breaker
- Live paper trading with alpaca
- Integrated dry-run mechanism
- Protection from duplicate orders with a pending order guard
- Automated daily execution via cron
- Visualising trading history and analysis (given enough time)
- 36 individual test functions to ensure correct implementation

## KEY DESIGN DECISIONS
- The bot runs daily between at times 13:31, 13:36 and 13:41, 14:31, 14:36 and 14:41 UK time, due to the markets being in New-york (account for time difference) and to account for timezones changing often on different dates.
- The bot runs daily, mostly silently, due to no rebalance being needed. A rebalance is performed every 21 sucessful runs (21 trading days), as per the strategy. The only time a notification is recieved is when either an order is pending, successful or something goes wrong.
- Plan was to increase the breadth of tickers we could trade, but when testing momentum signal on all S&P 500 tickers, results came back insignificant. 'Universe' of available tickers to trade is hence limited to the selection which yielded statistical significance with momentum strategy. For more details see ml_trading_signals README/notebook.
Why the centralised path-setup pattern exists across the repo.
- New importing script introduced due to automation system. Previous iterations (with various sys.path.append) caused importing errors when used in run_bot_scheduled due to different working directories, hence paths pointed nowhere. New script written to fit across portfolio and has been inserted at the start of each project with a note.

## RESULTS
To come given enough time to see meaningful results

## LIMITATIONS
- Bot currently only implements one strategy (momentum) on a selection of tickers. Ideally would like to implement more strategies on a wider range of assets/equities, which is the plan eventually
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
        pipeline.py — momentum selection, rebalance decisions, dry-run previews, bot control and summaries
        broker.py — Alpaca clients, order planning, pending-state persistence, submission, reconciliation and completion
        monitoring.py — formatting and sending Discord run-status alerts
        reporting.py — loading, validating and summarising completed rebalance history
        plotting_trading_bot.py — visualising completed rebalance and order history
        run_bot_scheduled.py — weekly cron entry point for running the bot and sending alerts
    notebooks/
        trading_bot.ipynb — main bot preview, dry-run, account and history analysis (to come)
    data/
        basket_state.json — completed basket, strategy and account state used for 21-session rebalancing
        pending_rebalance.json — in-progress execution journal, created only while a rebalance is unfinished
        rebalance_history/ — immutable JSON archives for completed rebalances
        bot_log.jsonl — legacy run log retained from the earlier reporting format
        cron.log — scheduled-run output
    tests/
        test_strategy_and_planning.py — testing to ensure the system actually enacts the correct 63d momentum strategy.
        test_preview_and_controller.py — testing to ensure our preview works correctly (builds orders but does not submit)
        test_rebalance_state_machine.py — ensuring all rebalancing tools work as intended
        test_monitoring.py — verifying whether discord alerts work as intended
        test_reporting.py — testing to ensure whether all reporting works correctly (whether things were successful/pending/unsucessful + why)

## USAGE
```python
from pathlib import Path
import pandas as pd
from alpaca.trading.requests import GetCalendarRequest

from pipeline import build_rebalance_preview, generate_trading_decisions, rebalance_summary, run_bot
from broker import get_client, check_order_status
from reporting import load_rebalance_history
from plotting_trading_bot import plot_rebalance_history

bot_root = _REPO_ROOT / "trading_bot"
env_path = bot_root / ".env"
client = get_client(env_path=str(env_path))

#use Alpaca's exchange calendar to identify the latest completed trading session
clock = client.get_clock()
now = pd.Timestamp(clock.timestamp)
if now.tzinfo is None:
    now = now.tz_localize("America/New_York")
else:
    now = now.tz_convert("America/New_York")

calendar = client.get_calendar(filters=GetCalendarRequest( start=(now - pd.Timedelta(days=400)).date(), end=now.date()))
trading_dates = pd.DatetimeIndex([pd.Timestamp(session.date).normalize() for session in calendar])
completed_dates = []
for date, session in zip(trading_dates, calendar):
    close = pd.Timestamp(session.close)
    if close.tzinfo is None:
        close = close.tz_localize("America/New_York")
    else:
        close = close.tz_convert("America/New_York")
    if close <= now:
        completed_dates.append(date)

signal_date = completed_dates[-1]

#get recommended trading decisions by the system
decisions, decision_plan = generate_trading_decisions(tickers=tickers, signal_date=signal_date, trading_dates=trading_dates, lookback_days=63, top_frac=0.20, rebalance_days=21, state_path=state_path)

#preview the next rebalance without submitting orders
preview = build_rebalance_preview(client=client, tickers=tickers, env_path=env_path, state_path=state_path, lookback_days=63, top_frac=0.20, rebalance_days=21)

#get decisions and corresponding orders together along with an alert. submit_orders=True makes the order happen. log to log_path
result = run_bot(tickers=tickers, submit_orders=False, env_path=env_path, state_path=state_path, lookback_days=63, top_frac=0.20, rebalance_days=21)

#see order status from alpaca
check_order_status(client)

#analyse past data run, and print a summary of last trades
rebalance_history, order_history = load_rebalance_history(history_path=history_path)
if not rebalance_history.empty:
    plot_rebalance_history(rebalance_history, order_history)
print(rebalance_summary(history_path=history_path))
```

automated scheduling is handled outside the notebook, in `run_bot_scheduled.py`
triggered daily by cron:
```
31,36,41 13,14 * * 1-5 /Users/oscarlewis/Desktop/python/.venv/bin/python -B /Users/oscarlewis/Desktop/python/trading_portfolio/trading_bot/src/run_bot_scheduled.py >> /Users/oscarlewis/Desktop/python/trading_portfolio/trading_bot/data/cron.log 2>&1
```