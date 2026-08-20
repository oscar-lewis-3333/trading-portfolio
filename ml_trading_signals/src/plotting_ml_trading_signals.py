import matplotlib.pyplot as plt
import numpy as np


def plot_trade_outcomes(pooled_df, model, feature_cols, test_dates, confidence_threshold=0.6):
#scatter every confident trade: predicted vs actual direction

    clean = pooled_df.dropna(subset=feature_cols + ['label'])
    test_mask = clean.index.isin(test_dates)
    X_test = clean.loc[test_mask, feature_cols]
    fwd_ret = clean.loc[test_mask, 'fwd_return']

    proba = model.predict_proba(X_test)[:, 1]
    confident = (proba >= confidence_threshold) | (proba <= 1 - confidence_threshold)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(proba[confident] >= 0.5, 'steelblue', 'coral')
    ax.scatter(range(confident.sum()), fwd_ret.values[confident]*100,
              c=colors, alpha=0.6, s=30)
    ax.axhline(0, color='black', linewidth=0.8)

    ax.scatter([], [], c='steelblue', label='Predicted buy')
    ax.scatter([], [], c='coral', label='Predicted sell')
    ax.set_xlabel('Trade index')
    ax.set_ylabel('Actual forward return (%)')
    ax.set_title(f'Confident trades — correct calls should sit\n'
                f'above 0 (blue) or below 0 (coral)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#part 2 
#we look at momentum sweep windows across lookback windowsand selection fractions, with significance marked
def plot_momentum_sweep(sweep_df):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for frac in sorted(sweep_df['top_frac'].unique()):
        subset = sweep_df[sweep_df['top_frac'] == frac].sort_values('lookback')
        lookback_days = subset['lookback'].str.extract(r'(\d+)').astype(int)[0]

        axes[0].plot(lookback_days, subset['top_minus_bottom'], 'o-',  label=f'top {int(frac*100)}%')

    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_xlabel('Lookback window (days)')
    axes[0].set_ylabel('Top − Bottom return (%)')
    axes[0].set_title('Momentum effect size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #right panel gives a heatmap-style significance scatter
    pivot = sweep_df.pivot(index='lookback', columns='top_frac', values='p_top_vs_bottom')
    pivot = pivot.reindex(['return_5d', 'return_10d', 'return_21d', 'return_63d'])

    im = axes[1].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels([f'{int(c*100)}%' for c in pivot.columns])
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set_xlabel('Top fraction selected')
    axes[1].set_title('p-value (green = significant)')
    plt.colorbar(im, ax=axes[1], label='p-value')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

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