## Crypto portfolio

Research concerning cryptocurrencies, with transaction costs based upon Revolut X offerings.

### Crypto absolute momentum 

**Selected for implementation**

Weekly absolute momentum on BTC, ETH against GBP. Each asset is held while its trailing 30d returns are positive, with cash being held otherwise. Over the 2024-2025 holdout, returned 96.50% after modelled transaction costs. Bootstrap confidence intervals include 0, so no statistically significant edge shown comparitively to its benchmark. Limited by only having a 2 asset universe, 2 year holdout and untested trading frequency.

[View project](crypto_momentum/)

### Crypto cross-sectional momentum

**Not selected for implementation**

Cross-sectional momentum tested on a current universe of cryptocurrencies which excludes memecoins, stablecoins and gold-backed coins. Sweep chose 90d momentum, top 10% as optimal on a training period. Rebelances occur weekly, with the strategy having a positive sharpe-difference of 0.347 compared to its benchmark over a testing period of 01/2024-08/2026. All Bootstrap confidence intervals contained 0, so no statistically significant edge shown. Both approaches lost money over the two years, with the strategy not planned to be implemented. Limited by untested trading frequecies and a relatively short training period.

[View project](crypto_cross_sectional/)

A crypto trading bot through Revolut X is coming soon, alongside further projects researching cryptocurrencies.