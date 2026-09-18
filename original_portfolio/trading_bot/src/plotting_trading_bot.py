import matplotlib.pyplot as plt
import pandas as pd


def plot_rebalance_history(rebalances, orders):
    if not isinstance(rebalances, pd.DataFrame) or not isinstance(orders, pd.DataFrame):
        raise TypeError("Rebalances and orders must be DataFrames")
    if rebalances.empty:
        raise ValueError("No completed rebalances to plot")

    required_rebalances = {
        'completed_at',
        'portfolio_value',
        'buy_notional',
        'sell_notional',
        'weighted_adverse_slippage_bps'
    }
    required_orders = {'completed_at', 'side', 'adverse_slippage_bps'}
    if required_rebalances.difference(rebalances.columns):
        raise ValueError("Rebalance history is missing required columns")
    if required_orders.difference(orders.columns):
        raise ValueError("Order history is missing required columns")

    history = rebalances.sort_values('completed_at').copy()
    history['completed_at'] = pd.to_datetime(history['completed_at'], utc=True)
    order_history = orders.copy()
    if not order_history.empty:
        order_history['completed_at'] = pd.to_datetime(order_history['completed_at'], utc=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(history['completed_at'], history['portfolio_value'], color='blue')
    axes[0].set_ylabel('Portfolio value ($)')
    axes[0].set_title('Completed Rebalance History')
    axes[0].grid(True)

    axes[1].plot(history['completed_at'], history['buy_notional'], color='green', label='Buys')
    axes[1].plot(history['completed_at'], history['sell_notional'], color='orange', label='Sells')
    axes[1].set_ylabel('Filled notional ($)')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].plot(history['completed_at'], history['weighted_adverse_slippage_bps'],color='red', label='Weighted mean')
    for side, colour in [('buy', 'green'), ('sell', 'orange')]:
        side_orders = order_history.loc[order_history['side'] == side]
        if not side_orders.empty:
            axes[2].scatter(side_orders['completed_at'], side_orders['adverse_slippage_bps'], color=colour, label=side.title())
    axes[2].set_ylabel('Adverse slippage (bps)')
    axes[2].set_xlabel('Rebalance completion')
    axes[2].legend()
    axes[2].grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    return fig
