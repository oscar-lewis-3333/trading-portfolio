import matplotlib.pyplot as plt
import numpy as np


def plot_oos_top_frac(daily_results):
    # Visualise paired OOS selection results.
    # These are overlapping forward outcomes, not an equity curve.

    required = {
        'date',
        'n_universe',
        'n_selected',
        'top_return',
        'universe_return',
        'excess_return',
    }

    missing = required.difference(daily_results.columns)
    if missing:
        raise ValueError(f"Required plotting columns are missing: {sorted(missing)}")

    if daily_results.empty:
        raise ValueError("Daily OOS results are empty")

    if daily_results[list(required)].isna().any().any():
        raise ValueError("Daily OOS results contain missing values")

    if daily_results['date'].duplicated().any():
        raise ValueError("Duplicate signal dates found")

    numeric_cols = [
        'n_universe',
        'n_selected',
        'top_return',
        'universe_return',
        'excess_return',
    ]

    values = daily_results[numeric_cols].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Daily OOS results contain non-finite values")

    if (daily_results['n_selected'] <= 0).any() or (daily_results['n_selected'] > daily_results['n_universe']).any():
        raise ValueError("Invalid selection counts found")

    top_returns = daily_results['top_return'].to_numpy(dtype=float) * 100
    
    universe_returns = daily_results['universe_return'].to_numpy(dtype=float) * 100
    excess_returns = daily_results['excess_return'].to_numpy(dtype=float) * 100

    selected_fraction = (daily_results['n_selected'] / daily_results['n_universe']).mean()

    selected_label = f"Top {selected_fraction:.0%}"

    lower = min(top_returns.min(), universe_returns.min())
    upper = max(top_returns.max(), universe_returns.max())
    padding = max((upper - lower) * 0.05, 0.1)
    limits = [lower - padding, upper + padding]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(universe_returns, top_returns, color='blue')
    axes[0].plot(limits, limits, color='black',)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel('Universe forward return (%)')
    axes[0].set_ylabel(f'{selected_label} forward return (%)')
    axes[0].set_title('Paired OOS returns by signal date')
    axes[0].grid(True, alpha=0.3)

    mean_excess = excess_returns.mean()
    beat_rate = (excess_returns > 0).mean()

    axes[1].hist(excess_returns, bins=30, color='blue')
    axes[1].axvline(0,color='black')
    axes[1].axvline(mean_excess, color='orange', label=f'Mean: {mean_excess:.2f}%',)
    axes[1].set_xlabel(f'{selected_label} minus universe return (%)')
    axes[1].set_ylabel('Number of signal dates')
    axes[1].set_title(f'Excess-return distribution\n' f'Selected group beat universe: {beat_rate:.1%}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig

#part 2

#show top-vs-bottom and top-vs-universe as separate figures so their properties are clear to see 
def plot_momentum_sweep(sweep_df):

    required = {'lookback', 'top_frac', 'top_minus_bottom', 'p_top_vs_bottom', 'top_minus_universe', 'p_top_vs_universe',}
    missing = required.difference(sweep_df.columns)
    if missing:
        raise ValueError(f"Momentum sweep columns are missing: {sorted(missing)}")

    lookback_order = ['return_5d', 'return_10d', 'return_21d', 'return_63d']

    def plot_comparison(effect_col, p_col, comparison_label):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for frac in sorted(sweep_df['top_frac'].unique()):
            subset = sweep_df[sweep_df['top_frac'] == frac].copy()
            subset['lookback_days'] = subset['lookback'].str.extract(r'(\d+)')[0].astype(int)
            subset = subset.sort_values('lookback_days')

            axes[0].plot(subset['lookback_days'], subset[effect_col],label=f'top {int(frac * 100)}%',)

        axes[0].axhline(0, color='black', linewidth=0.8)
        axes[0].set_xlabel('Lookback window (days)')
        axes[0].set_ylabel(f'{comparison_label} return (%)')
        axes[0].set_title(f'{comparison_label} effect size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        pivot = sweep_df.pivot(index='lookback', columns='top_frac', values=p_col)
        pivot = pivot.reindex(lookback_order)

        im = axes[1].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        axes[1].set_xticks(range(len(pivot.columns)))
        axes[1].set_xticklabels([f'{int(c * 100)}%' for c in pivot.columns])
        axes[1].set_yticks(range(len(pivot.index)))
        axes[1].set_yticklabels(pivot.index)
        axes[1].set_xlabel('Top fraction selected')
        axes[1].set_title(f'{comparison_label} Holm-adjusted p-value')
        fig.colorbar(im, ax=axes[1], label='p-value')

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                label = 'nan' if np.isnan(val) else f'{val:.3f}'
                axes[1].text(j, i, label, ha='center', va='center')

        fig.tight_layout()
        return fig

    bottom_figure = plot_comparison(
        'top_minus_bottom',
        'p_top_vs_bottom',
        'Top vs Bottom',
    )
    universe_figure = plot_comparison(
        'top_minus_universe',
        'p_top_vs_universe',
        'Top vs Universe',
    )
    plt.show()

    return bottom_figure, universe_figure

def plot_forward_test_setup(log_df):
    #we visualise the forward test at its starting point, stocks which are in top/bottom quartile.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    colors = {'top_quintile': 'deepskyblue', 'bottom_quintile': 'orangered', 'middle': 'lightgray'}
    log_sorted = log_df.sort_values('composite_rank')

    #left panel gives PE vs ROE scatter
    for group, color in colors.items():
        subset = log_sorted[log_sorted['group'] == group]
        axes[0].scatter(subset['pe_ratio'], subset['roe']*100, c=color, s=80, alpha=0.8, edgecolor='black', linewidth=0.5, label=group.replace('_', ' '))
        for _, row in subset.iterrows():
            axes[0].annotate(row['ticker'], (row['pe_ratio'], row['roe']*100),  xytext=(5, 5), textcoords='offset points', fontsize=8)

    axes[0].set_xlabel('P/E ratio (lower = cheaper)')
    axes[0].set_ylabel('ROE % (higher = more profitable)')
    axes[0].set_title('Starting position: value vs quality')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #right panel gives composite rank pie chart, which is what we have ordered by
    bar_colors = [colors[g] for g in log_sorted['group']]
    axes[1].barh(log_sorted['ticker'], log_sorted['composite_rank'], color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Composite rank (lower = better value+quality)')
    axes[1].set_title(f'Full ranking as of {log_df["start_date"].iloc[0]}')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    print(f"\nAs time progresses we predict that blue (top quintile) should on average outperform orange/red (bottom quintile). Reapply function with forward_evaluation_tracking")
