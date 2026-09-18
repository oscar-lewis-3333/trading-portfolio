import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_garch_forecast(df, vol_forecast, realised_vol, ticker, windows=(10,30), zoom_days=None):
    #aim to plot GARCH for the future with realised volatility as a reference line

    #we also plot the 10, 30 day rolling volatility for the ticker as a reference for GARCH.
    #this can be difficult to see, so we implement zoom_days, where you zoom in on the last zoom_days to better see whats happening
    plot_df = df.tail(zoom_days) if zoom_days else df

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['steelblue', 'purple']
    for i, window in enumerate(windows):
        rolling_vol = df['Return'].rolling(window).std() * np.sqrt(252) #annualised volatilty as seen in other files and previous projects
        rolling_vol = rolling_vol.tail(zoom_days) if zoom_days else rolling_vol #zoom_days is only activated if called, else normal
        ax.plot(plot_df.index, rolling_vol, color=colors[i], linewidth=1.2,
                label=f'{window}-day rolling volatility')

    forecast_dates = pd.bdate_range(start=df.index[-1], periods=len(vol_forecast)+1)[1:] #generates business days 'into the future' so can see forecast in action
    ax.plot(forecast_dates, vol_forecast, color='coral', linewidth=2,
            marker='o', markersize=5, label='GARCH forecast')

    ax.axhline(realised_vol, color='gray', linestyle='--', linewidth=1,
               label=f'Full-period realised vol ({realised_vol*100:.1f}%)')

    title_suffix = f' (last {zoom_days} days)' if zoom_days else '' #normal if zoom_days not called, else we add zoom_days
    ax.set_title(f'{ticker} — Volatility: Historical vs GARCH Forecast{title_suffix}')
    ax.set_ylabel('Annualised volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
