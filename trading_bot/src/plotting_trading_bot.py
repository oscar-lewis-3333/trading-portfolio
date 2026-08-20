import matplotlib.pyplot as plt
import pandas as pd

def plot_run_history(log_path):
    from broker import load_run_history

    history = load_run_history(log_path=log_path)
    if history.empty:
        print("No History Yet")
        return

    history['timestamp'] = pd.to_datetime(history['timestamp'])
    history['portfolio_value'] = history['account'].apply(lambda a: a['portfolio_value'])

    fig, axes = plt.subplots(3, 1)
    #the values of the account over time
    axes[0].plot(history['timestamp'], history['portfolio_value'], color='blue')
    axes[0].set_ylabel('Portfolio value ($)')
    axes[0].set_title('Trading Bot History')
    axes[0].grid(True)

    #exposure of portfolio over time

    axes[1].plot(history['timestamp'], history['circuit_breaker_exposure'], color='green')
    axes[1].set_ylabel('Exposure')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True)

    #max drawdown of portfolio over time

    axes[2].plot(history['timestamp'], history['current_drawdown']*100, color='red')
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].set_xlabel('Run date')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

