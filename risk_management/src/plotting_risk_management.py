import matplotlib.pyplot as plt

def plot_equity_drawdown_exposure(equity, managed_equity, drawdown, exposure):
      
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    #first tab shows managed/unmanaged momentum strategy over the past 10 years
    axes[0].plot(equity.index, equity['equity'].values, color='blue', label='Raw strategy') #when plotting first time, plt was plotting portfolio return too, so have to specify
    axes[0].plot(managed_equity.index, managed_equity.values, color='red', label='Managed strategy')
    axes[0].set_ylabel('Equity (growth of $1)')
    axes[0].set_xlabel('Date (years)')
    axes[0].set_title('Momentum strategy — daily equity curve')
    axes[0].legend()
    axes[0].grid(True)


    #shows max-drawdown of the raw strategy over the past 10 years
    axes[1].fill_between(drawdown.index, drawdown.values*100, 0, color='coral', alpha=0.6)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Date(years)')
    axes[1].axhline(-10, color='orange', label='Soft limit')
    axes[1].axhline(-25, color='red', label='Hard limit')
    axes[1].legend()
    axes[1].grid(True)

    #shows exposure of the strategy over the past 10 years
    axes[2].plot(drawdown.index, exposure, color='green')
    axes[2].fill_between(drawdown.index, exposure, 0, color='green', alpha=0.2)
    axes[2].set_ylabel('Exposure scale')
    axes[2].set_xlabel('Date (years)')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()



def plot_position_sizing_comparison(kelly_size, vol_size, cvar_size, final_size):
    #plot the sizing of the methods against one another to see the comparison
    methods = ['Full allocation\n(no sizing)', 'Kelly\n(quarter)', 'Volatility-\nscaled', 'CVaR-\nconstrained', 'Final\n(minimum)']
    sizes = [1.0, kelly_size, vol_size, cvar_size, final_size]
    colors = ['gray', 'blue', 'blue', 'blue', 'red']

    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(methods, sizes, color=colors, edgecolor='black')

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{size*100:.1f}%', ha='center', fontsize=10)

    ax.axhline(1.0, color='gray')
    ax.set_ylabel('Position size (fraction of capital)')
    ax.set_title('Position sizing methods compared')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_three_way_comparison(raw_equity, drawdown_managed_equity, regime_managed_equity):
    #compare three strategies - standard momentum, drawdown momentum, regime momentum
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(raw_equity.index, raw_equity.values, color='gray', label='Raw (no risk management)')
    ax.plot(drawdown_managed_equity.index, drawdown_managed_equity.values, color='blue', label='Drawdown circuit breaker')
    ax.plot(regime_managed_equity.index, regime_managed_equity.values, color='red', label='Volatility regime scaling')

    ax.set_ylabel('Equity (growth of $1')
    ax.set_xlabel('Date')
    ax.set_title('Risk management approaches compared')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_risk_methods(raw_equity, drawdown_managed, regime_managed, stopped_equity):

    #show all risk methods on the same plot for completness
    fig, ax = plt.subplots(figsize=(12,6))

    series = [
        (raw_equity, 'gray', 'Raw (no risk management)'),
        (drawdown_managed, 'blue', 'Drawdown circuit breaker'),
        (regime_managed, 'red', 'Volatility regime scaling'),
        (stopped_equity, 'purple', 'Position-level stops'),
    ]

    for eq, color, label in series:
        ax.plot(eq.index, eq.values, color=color, label=label)

    ax.set_ylabel('Equity (growth of $1)')
    ax.set_xlabel('Date')
    ax.set_title('All risk management methods compared')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()