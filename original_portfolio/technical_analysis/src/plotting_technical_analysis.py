#this python file is dedicated to writing code for professional looking plot

import matplotlib.pyplot as plt

def plot_signals(df, ticker):
    #we plot the close prices of the ticker on the same plot as the SMA_50, SMA_200

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    axes[0].plot(df.index, df['Close'], color='black', linewidth=1, label='Price')
    axes[0].plot(df.index, df['SMA_50'], color='blue', linewidth=1,
                 linestyle='--', label='SMA 50', alpha=0.7)
    axes[0].plot(df.index, df['SMA_200'], color='red', linewidth=1,
                 linestyle='--', label='SMA 200', alpha=0.7)

    buys = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]
    axes[0].scatter(buys.index, buys['Close'], marker='^', color='green',
                    s=150, zorder=5, label='Buy signal')
    axes[0].scatter(sells.index, sells['Close'], marker='v', color='red',
                    s=150, zorder=5, label='Sell signal')

    axes[0].set_title(f'{ticker} — Signals')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #we plot the ADX of the ticker beneath this, which measures the strength of the trend

    axes[1].plot(df.index, df['ADX'], color='orange', linewidth=1.2)
    axes[1].axhline(30, color='red', linestyle='--', linewidth=1, label='ADX 30')
    axes[1].set_ylabel('ADX')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df.index, df['RSI'], color='purple', linewidth=1.2)
    axes[2].axhline(70, color='red', linestyle='--', linewidth=1)
    axes[2].axhline(30, color='green', linestyle='--', linewidth=1)
    axes[2].set_ylabel('RSI')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#we plot the backtest signal, i.e, the buy and hold strategy against our buy/sell signal strategy

def plot_backtest(df, ticker):

    #the first panel is dedicated to the value of our buy/sell strategy
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [2, 1]})

    axes[0].plot(df.index, df['Portfolio_Value'], color='green',
                 linewidth=1.5, label='Strategy')
    axes[0].plot(df.index, df['Buy_Hold_Value'], color='gray',
                 linewidth=1.5, linestyle='--', label='Buy & Hold')
    axes[0].set_title(f'{ticker} — Strategy vs Buy & Hold')
    axes[0].set_ylabel('Portfolio Value (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #second panel is dedicated to the difference between the buy/sell strategy and the buy and hold strategy
    
    outperformance = df['Portfolio_Value'] - df['Buy_Hold_Value']
    colors = ['green' if v >= 0 else 'red' for v in outperformance]
    axes[1].bar(df.index, outperformance, color=colors, alpha=0.6, width=1)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Strategy Outperformance vs Buy & Hold')
    axes[1].set_ylabel('$ Difference')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()