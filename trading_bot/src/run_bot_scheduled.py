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

import sys
import os

from pipeline import run_bot


#universe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "XOM", "WMT", "PG", "HD", "DIS", "NFLX", "AMD", "INTC", "CSCO", "ADBE", "CRM"]


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)

    try:
        run_bot(tickers, submit_orders=True, log_path=os.path.join(base, 'data', 'bot_log.jsonl'), env_path=os.path.join(base, '.env'),)
    except Exception as e:
        from monitoring import send_alert
        import traceback
        send_alert(subject="Trading Bot Failed",body=f"Run failed. \n\n{traceback.format_exc()}", env_path=os.path.join(base,'.env'),)
        raise


