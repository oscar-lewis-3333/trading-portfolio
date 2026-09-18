"""Print development-only cost diagnostics using the independent reference engine."""
import json
import math
import statistics
import sys

sys.dont_write_bytecode = True

from _support import development_prices, reference_equity, reference_targets


if __name__ == "__main__":
    prices = development_prices()
    result = {}
    for mode in ("momentum", "buy_and_hold", "weekly"):
        targets = reference_targets(prices, mode)
        net = reference_equity(prices, targets, rate=0.00125)
        gross = reference_equity(prices, targets, rate=0)
        previous = net.shift(1)
        previous.iloc[0] = 1000.0
        returns = (net / previous - 1).tolist()
        result[mode] = {
            "final_equity_with_costs": float(net.iloc[-1]),
            "final_equity_zero_costs": float(gross.iloc[-1]),
            "annualised_arithmetic_mean_return": statistics.mean(returns) * 365.25,
            "annualised_volatility": statistics.stdev(returns) * math.sqrt(365.25),
        }
    print(json.dumps(result, indent=2))
