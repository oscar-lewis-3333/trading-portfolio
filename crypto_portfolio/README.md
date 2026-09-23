# Crypto portfolio

Research concerning cryptocurrencies, with transaction costs based upon Revolut X offerings.

## Crypto absolute momentum 

**Selected for implementation**

Weekly absolute momentum on BTC, ETH against GBP. Each asset is held while its trailing 30d returns are positive, with cash being held otherwise. Over the 2024-2025 holdout, returned 96.50% after modelled transaction costs. Bootstrap confidence intervals include 0, so no statistically significant edge shown comparitively to its benchmark. Limited by only having a 2 asset universe, 2 year holdout and untested trading frequency.

[View project](crypto_momentum/)

## Crypto cross-sectional momentum

**Not selected for implementation**

Cross-sectional momentum tested on a current universe of cryptocurrencies which excludes memecoins, stablecoins and gold-backed coins. Sweep chose 90d momentum, top 10% as optimal on a training period. Rebelances occur weekly, with the strategy having a positive sharpe-difference of 0.347 compared to its benchmark over a testing period of 01/2024-08/2026. All Bootstrap confidence intervals contained 0, so no statistically significant edge shown. Both approaches lost money over the two years, with the strategy not planned to be implemented. Limited by untested trading frequecies and a relatively short training period.

### Bitcoin Filter

**Selected for implementation**

Implement the same strategy as above, but only invest when Bitcoin's 120d trailing return is positive, with full cash held otherwise. Returned 36.29% after modelled transaction costs over the same test period as above, with a max drawdown of 64.17%. Bootstrap significance testing support a Sharpe advantage over its filtered, equal-weight counterpart, but inconclusive over the filters advantage against the non-filter results, despite promising results. Same limitations apply as above.

[View project](crypto_cross_sectional/)

## Crypto reversal signals

**Not selected for implementation**

Cross-sectional reversal signals approach tested by buying recent relative losers from the eligible crypto universe. Parameter sweep across 60 configuration on a development period. On a validation period of 2023-2024, strategy returned 155.57% after modelled transaction fees of 25bp round trip, as opposed to its benchmark's 293.46% returns. The approach also had a higher volatility and worse drawdown. Stopped after this due to the weak performance relative to the benchmark. No significance test was ran, and the reserved holdout was not used.

[View project](crypto_reversal/)

A crypto trading bot through Revolut X is coming soon, alongside further projects researching cryptocurrencies.