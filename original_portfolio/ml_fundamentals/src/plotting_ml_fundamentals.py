import numpy as np
import matplotlib.pyplot as plt


def plot_complexity_sweep(sweep_df, clip_at=-1):
    #we visualise the bias-variance tradeoff from the polynomial complexity sweep

    #clip_at: test R^2 lower bound for main plot. Second panel shows true magnitude on symlog scale
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    degrees = sweep_df.index

    # Panel 1 — clipped, showing where the optimum actually is
    axes[0].plot(degrees, sweep_df['train_r2'], 'o-', color='steelblue',
                 linewidth=2, label='Training R²')
    axes[0].plot(degrees, sweep_df['test_r2'].clip(lower=clip_at), 'o-',
                 color='coral', linewidth=2, label='Test R² (clipped)')
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--',
                    label='R² = 0 (predicting the mean)')

    best = sweep_df['test_r2'].idxmax()
    axes[0].axvline(best, color='green', linestyle=':', linewidth=1.5,
                    label=f'Best test R² (degree {best})')

    axes[0].set_xlabel('Polynomial degree')
    axes[0].set_ylabel('R²')
    axes[0].set_title('Bias-variance tradeoff')
    axes[0].set_xticks(degrees)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — true magnitude of the divergence
    axes[1].plot(degrees, sweep_df['train_r2'], 'o-', color='steelblue',
                 linewidth=2, label='Training R²')
    axes[1].plot(degrees, sweep_df['test_r2'], 'o-', color='coral',
                 linewidth=2, label='Test R² (true scale)')
    axes[1].set_yscale('symlog')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')

    axes[1].set_xlabel('Polynomial degree')
    axes[1].set_ylabel('R² (symlog scale)')
    axes[1].set_title('True scale of test-set collapse')
    axes[1].set_xticks(degrees)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()