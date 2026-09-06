import json
import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd


REBALANCE_COLUMNS = [
    'rebalance_id',
    'signal_date',
    'execution_date',
    'completed_at',
    'account_id',
    'lookback_days',
    'top_frac',
    'rebalance_days',
    'universe_size',
    'weighting',
    'tickers',
    'positions',
    'n_positions',
    'n_orders',
    'n_buys',
    'n_sells',
    'buy_notional',
    'sell_notional',
    'gross_traded_notional',
    'net_cash_flow',
    'weighted_adverse_slippage_bps',
    'cash_reserve_fraction',
    'cash',
    'equity',
    'portfolio_value',
    'invested_fraction',
]

ORDER_COLUMNS = [
    'rebalance_id',
    'completed_at',
    'ticker',
    'side',
    'planned_qty',
    'filled_qty',
    'planned_reference_price',
    'submission_reference_price',
    'submission_quote_at',
    'filled_avg_price',
    'reference_notional',
    'filled_notional',
    'adverse_shortfall',
    'adverse_slippage_bps',
    'client_order_id',
    'broker_order_id',
    'submission_attempted_at',
]


def _number(value, field, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error

    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{field} is invalid")
    return number


def _positive_integer(value, field):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _position_map(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dictionary")

    positions = {}
    for ticker, quantity in value.items():
        if not isinstance(ticker, str) or not ticker:
            raise ValueError(f"{field} contains an invalid ticker")
        positions[ticker] = _number(
            quantity, f"{field} quantity for {ticker}", positive=True
        )
    return positions


def _session_date(value, field):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error

    if (pd.isna(timestamp) or timestamp.tzinfo is not None or
            timestamp != timestamp.normalize()):
        raise ValueError(f"{field} is invalid")
    return timestamp


def _utc_timestamp(value, field):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error

    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert('UTC')


def _empty_history():
    return (
        pd.DataFrame(columns=REBALANCE_COLUMNS),
        pd.DataFrame(columns=ORDER_COLUMNS),
    )


def load_rebalance_history(history_path=None):
    history_dir = (
        Path(history_path).expanduser().resolve()
        if history_path is not None
        else Path(__file__).resolve().parents[1] / 'data' / 'rebalance_history'
    )

    if not history_dir.exists():
        return _empty_history()
    if not history_dir.is_dir():
        raise ValueError("Rebalance history path must be a directory")

    rebalance_rows = []
    order_rows = []

    for archive_path in sorted(history_dir.glob('*.json')):
        try:
            with archive_path.open(encoding='utf-8') as file:
                archive = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read rebalance archive: {archive_path.name}"
            ) from error

        try:
            if (not isinstance(archive, dict) or
                    archive.get('schema_version') != 2 or
                    archive.get('status') != 'completed'):
                raise ValueError("expected a completed schema-version-2 record")

            rebalance_id = archive.get('rebalance_id')
            if not isinstance(rebalance_id, str) or not rebalance_id:
                raise ValueError("invalid rebalance ID")
            try:
                canonical_id = UUID(rebalance_id).hex
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError("invalid rebalance ID") from error
            if canonical_id != rebalance_id or archive_path.stem != rebalance_id:
                raise ValueError("archive filename does not match its rebalance ID")

            account_id = archive.get('account_id')
            if not isinstance(account_id, str) or not account_id.strip():
                raise ValueError("invalid account ID")

            completed_state = archive.get('completed_state')
            proposed_state = archive.get('proposed_state')
            if not isinstance(completed_state, dict):
                raise ValueError("missing completed state")
            if not isinstance(proposed_state, dict):
                raise ValueError("missing proposed state")

            if (completed_state.get('schema_version') != 2 or
                    proposed_state.get('schema_version') != 2):
                raise ValueError("invalid state schema version")
            if (completed_state.get('rebalance_id') != rebalance_id or
                    completed_state.get('account_id') != account_id):
                raise ValueError("completed-state identity mismatch")

            signal_date = _session_date(archive.get('signal_date'), 'Signal date')
            execution_date = _session_date(
                archive.get('execution_date'), 'Execution date'
            )
            execution_open = _utc_timestamp(
                archive.get('execution_open'), 'Execution open'
            ).tz_convert('America/New_York')
            execution_window_end = _utc_timestamp(
                archive.get('execution_window_end'), 'Execution window end'
            ).tz_convert('America/New_York')
            if execution_date <= signal_date:
                raise ValueError("execution date must follow signal date")
            if (execution_open.date() != execution_date.date() or
                    execution_window_end.date() != execution_date.date() or
                    execution_window_end <= execution_open):
                raise ValueError("invalid execution window")
            if (_session_date(completed_state.get('selection_date'), 'Selection date') !=
                    signal_date or
                    _session_date(proposed_state.get('selection_date'), 'Selection date') !=
                    signal_date or
                    _session_date(completed_state.get('planned_execution_date'),
                                  'Planned execution date') != execution_date):
                raise ValueError("archive dates disagree")

            completed_at = _utc_timestamp(
                completed_state.get('completed_at'), 'Completion timestamp'
            )
            archive_completed_at = _utc_timestamp(
                archive.get('completed_at'), 'Archive completion timestamp'
            )
            if archive_completed_at != completed_at:
                raise ValueError("completion timestamps disagree")
            if completed_at < execution_open.tz_convert('UTC'):
                raise ValueError("completion precedes the execution window")

            strategy = completed_state.get('strategy')
            if (not isinstance(strategy, dict) or
                    proposed_state.get('strategy') != strategy):
                raise ValueError("strategy configuration mismatch")

            lookback_days = _positive_integer(
                strategy.get('lookback_days'), 'Lookback days'
            )
            rebalance_days = _positive_integer(
                strategy.get('rebalance_days'), 'Rebalance days'
            )
            top_frac = _number(strategy.get('top_frac'), 'Top fraction', positive=True)
            if top_frac > 1:
                raise ValueError("top fraction cannot exceed one")

            universe = strategy.get('universe')
            if (not isinstance(universe, list) or len(universe) < 5 or
                    any(not isinstance(ticker, str) or not ticker for ticker in universe) or
                    len(universe) != len(set(universe))):
                raise ValueError("invalid strategy universe")

            weighting = strategy.get('weighting')
            if weighting != 'equal_at_rebalance':
                raise ValueError("unsupported weighting method")

            tickers = completed_state.get('tickers')
            if (not isinstance(tickers, list) or not tickers or
                    any(not isinstance(ticker, str) or not ticker for ticker in tickers) or
                    len(tickers) != len(set(tickers)) or
                    not set(tickers).issubset(universe) or
                    proposed_state.get('tickers') != tickers):
                raise ValueError("invalid completed holdings")

            expected_count = max(1, int(len(universe) * top_frac))
            if len(tickers) != expected_count:
                raise ValueError("basket size disagrees with the strategy")

            starting_positions = _position_map(
                archive.get('starting_positions'), 'Starting positions'
            )
            expected_positions = _position_map(
                archive.get('expected_positions'), 'Expected positions'
            )
            clean_positions = _position_map(
                completed_state.get('positions_at_completion'),
                'Completed positions',
            )
            if set(starting_positions).difference(universe):
                raise ValueError("starting positions lie outside the strategy universe")
            if (set(expected_positions) != set(tickers) or
                    set(clean_positions) != set(tickers)):
                raise ValueError("completed holdings do not match the selected basket")
            if any(
                not math.isclose(
                    clean_positions[ticker], expected_positions[ticker],
                    rel_tol=0.0, abs_tol=1e-9,
                )
                for ticker in tickers
            ):
                raise ValueError("completed positions differ from expected positions")

            execution_policy = completed_state.get('execution_policy')
            if (not isinstance(execution_policy, dict) or
                    archive.get('execution_policy') != execution_policy):
                raise ValueError("execution-policy mismatch")
            cash_reserve_fraction = _number(
                execution_policy.get('cash_reserve_fraction'),
                'Cash reserve fraction',
            )
            if not 0 <= cash_reserve_fraction < 1:
                raise ValueError("invalid cash reserve fraction")
            execution_window_minutes = _positive_integer(
                execution_policy.get('execution_window_minutes'),
                'Execution window minutes',
            )
            price_max_age_seconds = _number(
                execution_policy.get('price_max_age_seconds'),
                'Maximum price age',
                positive=True,
            )
            if (execution_policy.get('order_type') != 'market' or
                    execution_policy.get('time_in_force') != 'day' or
                    execution_policy.get('extended_hours') is not False):
                raise ValueError("unsupported execution policy")
            if (execution_window_end > execution_open +
                    pd.Timedelta(minutes=execution_window_minutes)):
                raise ValueError("execution window conflicts with its policy")

            snapshot = completed_state.get('account_snapshot')
            if (not isinstance(snapshot, dict) or
                    {'cash', 'equity', 'portfolio_value'}.difference(snapshot)):
                raise ValueError("invalid account snapshot")
            cash = _number(snapshot['cash'], 'Cash')
            equity = _number(snapshot['equity'], 'Equity', positive=True)
            portfolio_value = _number(
                snapshot['portfolio_value'], 'Portfolio value', positive=True
            )
            invested_fraction = 1.0 - cash / portfolio_value

            orders = archive.get('orders')
            if not isinstance(orders, list):
                raise ValueError("orders must be a list")

            seen_client_ids = set()
            seen_order_tickers = set()
            archive_order_rows = []

            for order in orders:
                if (not isinstance(order, dict) or order.get('status') != 'filled' or
                        order.get('side') not in {'buy', 'sell'}):
                    raise ValueError("archive contains an incomplete order")

                ticker = order.get('ticker')
                if (not isinstance(ticker, str) or ticker not in universe or
                        ticker in seen_order_tickers):
                    raise ValueError("invalid or duplicate order ticker")
                seen_order_tickers.add(ticker)

                client_order_id = order.get('client_order_id')
                broker_order_id = order.get('broker_order_id')
                if (not isinstance(client_order_id, str) or not client_order_id or
                        client_order_id in seen_client_ids or
                        not isinstance(broker_order_id, str) or not broker_order_id):
                    raise ValueError("invalid order identifier")
                seen_client_ids.add(client_order_id)

                planned_qty = _number(order.get('qty'), 'Planned quantity', positive=True)
                filled_qty = _number(order.get('filled_qty'), 'Filled quantity', positive=True)
                if not math.isclose(
                    planned_qty, filled_qty, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError("filled quantity differs from planned quantity")

                planned_reference_price = _number(
                    order.get('reference_price'), 'Reference price', positive=True
                )
                submission_reference_price = _number(
                    order.get('submission_reference_price'),
                    'Submission reference price',
                    positive=True,
                )
                submission_quote_at = _utc_timestamp(
                    order.get('submission_quote_at'), 'Submission quote timestamp'
                )
                filled_avg_price = _number(
                    order.get('filled_avg_price'), 'Average fill price', positive=True
                )
                attempted_at = _utc_timestamp(
                    order.get('submission_attempted_at'), 'Submission timestamp'
                )
                execution_open_utc = execution_open.tz_convert('UTC')
                execution_window_end_utc = execution_window_end.tz_convert('UTC')
                if not (execution_open_utc <= submission_quote_at <= attempted_at <
                        execution_window_end_utc):
                    raise ValueError("order timestamps conflict with the execution window")
                if attempted_at > completed_at:
                    raise ValueError("submission timestamp follows completion")
                if ((attempted_at - submission_quote_at).total_seconds() >
                        price_max_age_seconds):
                    raise ValueError("submission quote exceeds its maximum age")

                reference_notional = filled_qty * submission_reference_price
                filled_notional = filled_qty * filled_avg_price
                if order['side'] == 'buy':
                    adverse_shortfall = (
                        filled_avg_price - submission_reference_price
                    ) * filled_qty
                else:
                    adverse_shortfall = (
                        submission_reference_price - filled_avg_price
                    ) * filled_qty
                adverse_slippage_bps = (
                    adverse_shortfall / reference_notional
                ) * 10_000

                archive_order_rows.append({
                    'rebalance_id': rebalance_id,
                    'completed_at': completed_at,
                    'ticker': ticker,
                    'side': order['side'],
                    'planned_qty': planned_qty,
                    'filled_qty': filled_qty,
                    'planned_reference_price': planned_reference_price,
                    'submission_reference_price': submission_reference_price,
                    'submission_quote_at': submission_quote_at,
                    'filled_avg_price': filled_avg_price,
                    'reference_notional': reference_notional,
                    'filled_notional': filled_notional,
                    'adverse_shortfall': adverse_shortfall,
                    'adverse_slippage_bps': adverse_slippage_bps,
                    'client_order_id': client_order_id,
                    'broker_order_id': broker_order_id,
                    'submission_attempted_at': attempted_at,
                })

            if archive_order_rows:
                if archive.get('no_order_confirmed_at') is not None:
                    raise ValueError("an ordered rebalance has a no-order confirmation")
            else:
                no_order_confirmed_at = _utc_timestamp(
                    archive.get('no_order_confirmed_at'),
                    'No-order confirmation timestamp',
                )
                if not (execution_open.tz_convert('UTC') <= no_order_confirmed_at <
                        execution_window_end.tz_convert('UTC') and
                        no_order_confirmed_at <= completed_at):
                    raise ValueError("no-order confirmation lies outside its execution window")

            reconciled_positions = starting_positions.copy()
            for order in archive_order_rows:
                signed_quantity = (
                    order['filled_qty']
                    if order['side'] == 'buy'
                    else -order['filled_qty']
                )
                final_quantity = reconciled_positions.get(order['ticker'], 0.0) + signed_quantity
                if final_quantity < -1e-9:
                    raise ValueError("filled orders produce a negative position")
                if abs(final_quantity) <= 1e-9:
                    reconciled_positions.pop(order['ticker'], None)
                else:
                    reconciled_positions[order['ticker']] = final_quantity

            if (set(reconciled_positions) != set(expected_positions) or any(
                    not math.isclose(
                        reconciled_positions[ticker], expected_positions[ticker],
                        rel_tol=0.0, abs_tol=1e-9,
                    )
                    for ticker in expected_positions
            )):
                raise ValueError("filled orders do not reconcile to expected positions")

            buy_notional = sum(
                order['filled_notional'] for order in archive_order_rows
                if order['side'] == 'buy'
            )
            sell_notional = sum(
                order['filled_notional'] for order in archive_order_rows
                if order['side'] == 'sell'
            )
            gross_notional = buy_notional + sell_notional
            gross_reference_notional = sum(
                order['reference_notional'] for order in archive_order_rows
            )
            weighted_slippage = (
                sum(order['adverse_shortfall'] for order in archive_order_rows) /
                gross_reference_notional * 10_000
                if gross_reference_notional > 0 else np.nan
            )

            rebalance_rows.append({
                'rebalance_id': rebalance_id,
                'signal_date': signal_date,
                'execution_date': execution_date,
                'completed_at': completed_at,
                'account_id': account_id,
                'lookback_days': lookback_days,
                'top_frac': top_frac,
                'rebalance_days': rebalance_days,
                'universe_size': len(universe),
                'weighting': weighting,
                'tickers': tuple(tickers),
                'positions': clean_positions,
                'n_positions': len(tickers),
                'n_orders': len(archive_order_rows),
                'n_buys': sum(order['side'] == 'buy' for order in archive_order_rows),
                'n_sells': sum(order['side'] == 'sell' for order in archive_order_rows),
                'buy_notional': buy_notional,
                'sell_notional': sell_notional,
                'gross_traded_notional': gross_notional,
                'net_cash_flow': sell_notional - buy_notional,
                'weighted_adverse_slippage_bps': weighted_slippage,
                'cash_reserve_fraction': cash_reserve_fraction,
                'cash': cash,
                'equity': equity,
                'portfolio_value': portfolio_value,
                'invested_fraction': invested_fraction,
            })
            order_rows.extend(archive_order_rows)

        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid completed rebalance archive {archive_path.name}: {error}"
            ) from error

    if not rebalance_rows:
        return _empty_history()

    rebalances = pd.DataFrame(rebalance_rows)
    rebalances = rebalances.sort_values(
        ['completed_at', 'rebalance_id']
    ).reset_index(drop=True)
    if rebalances['rebalance_id'].duplicated().any():
        raise ValueError("Duplicate rebalance IDs found")

    orders = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    if not orders.empty:
        orders = orders.sort_values(
            ['completed_at', 'rebalance_id', 'ticker']
        ).reset_index(drop=True)

    return rebalances[REBALANCE_COLUMNS], orders[ORDER_COLUMNS]


def latest_rebalance_summary(rebalances, orders):
    if not isinstance(rebalances, pd.DataFrame) or not isinstance(orders, pd.DataFrame):
        raise TypeError("Rebalances and orders must be DataFrames")
    if rebalances.empty:
        return "No completed rebalances"

    required_rebalances = set(REBALANCE_COLUMNS)
    required_orders = set(ORDER_COLUMNS)
    if required_rebalances.difference(rebalances.columns):
        raise ValueError("Rebalance history is missing required columns")
    if required_orders.difference(orders.columns):
        raise ValueError("Order history is missing required columns")

    latest = rebalances.sort_values(
        ['completed_at', 'rebalance_id']
    ).iloc[-1]
    latest_orders = orders.loc[orders['rebalance_id'] == latest['rebalance_id']]
    completed_at = pd.Timestamp(latest['completed_at']).tz_convert('UTC')
    basket = ', '.join(
        f"{ticker} ({latest['positions'][ticker]:g})" for ticker in latest['tickers']
    )

    lines = [
        f"Latest completed rebalance: {completed_at:%Y-%m-%d %H:%M UTC}",
        f"Signal/execution dates: {latest['signal_date'].date()} / "
        f"{latest['execution_date'].date()}",
        f"Strategy: {int(latest['lookback_days'])}-session momentum, "
        f"top {latest['top_frac'] * 100:g}%, rebalanced every "
        f"{int(latest['rebalance_days'])} sessions",
        f"Basket: {basket}",
        f"Execution: {int(latest['n_buys'])} buys, "
        f"{int(latest['n_sells'])} sells, "
        f"${latest['gross_traded_notional']:,.2f} gross notional",
    ]

    if math.isfinite(latest['portfolio_value']):
        lines.append(
            f"Account snapshot: ${latest['portfolio_value']:,.2f} portfolio value, "
            f"${latest['cash']:,.2f} cash, "
            f"{latest['invested_fraction'] * 100:.2f}% invested"
        )
    else:
        lines.append("Account snapshot: unavailable for this legacy archive")

    if not latest_orders.empty:
        lines.append(
            "Notional-weighted adverse slippage: "
            f"{latest['weighted_adverse_slippage_bps']:+.2f} bps"
        )

    return '\n'.join(lines)
