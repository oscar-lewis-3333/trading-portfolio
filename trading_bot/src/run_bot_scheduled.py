#script is necessaty here as it runs as its own process. This does not run when the associated notebook runs, but rather every monday by itself.
import sys
from pathlib import Path
if not globals().get("_TRADING_PORTFOLIO_PATHS_READY"):
    _HERE = (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else Path.cwd().resolve()
    )
    _PROJECTS = (
        "past_market_analysis",
        "technical_analysis",
        "options_pricing",
        "time_series_forecasting",
        "portfolio_construction",
        "ml_fundamentals",
        "ml_trading_signals",
        "risk_management",
        "trading_bot",
    )
    _REPO_ROOT = next(
        (
            directory
            for directory in (_HERE, *_HERE.parents)
            if all((directory / project).is_dir() for project in _PROJECTS)
        ),
        None,
    )
    if _REPO_ROOT is None:
        raise RuntimeError(
            "Could not locate the trading_portfolio repository root."
        )
    _SRC_PATHS = [
        str((_REPO_ROOT / project / "src").resolve())
        for project in _PROJECTS
    ]
    sys.path.extend(path for path in _SRC_PATHS if path not in sys.path)
    _TRADING_PORTFOLIO_PATHS_READY = True


import os
from datetime import time
from zoneinfo import ZoneInfo

from broker import get_client
from pipeline import run_bot
from monitoring import send_alert, send_run_alert

#universe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "XOM", "WMT", "PG", "HD", "DIS", "NFLX", "AMD", "INTC", "CSCO", "ADBE", "CRM"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)

    env_path = os.path.join(base, ".env")
    state_path = os.path.join(base, "data", "basket_state.json")

    try:
        pending_path = Path(base) / "data" / "pending_rebalance.json"

        if not pending_path.exists():
            clock = get_client(env_path=env_path).get_clock()
            broker_timestamp = clock.timestamp
            if broker_timestamp.tzinfo is None:
                raise ValueError("Broker timestamp must be timezone-aware")

            new_york_time = broker_timestamp.astimezone(ZoneInfo("America/New_York")).time()
            inside_execution_window = clock.is_open and time(9, 30) <= new_york_time < time(9, 45)            
            if not inside_execution_window:
                raise SystemExit(0)

        result = run_bot(tickers=tickers, submit_orders=False, env_path=env_path, state_path=state_path, lookback_days=63, top_frac=0.20, rebalance_days=21)
        send_run_alert(result=result, env_path=env_path)

    except Exception:
        import traceback
        send_alert(subject="Trading Bot Failed", body=f"Run failed.\n\n{traceback.format_exc()}", env_path=env_path)
        raise
