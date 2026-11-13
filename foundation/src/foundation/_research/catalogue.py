import pandas as pd

FAMILIES = ("trend", "cross_section", "positioning", "flow", "event", "microstructure", "volatility", "calendar", "macro", "attention")
IMPLEMENTATIONS = ("beta", "xs", "exec")
DATA_TIERS = ("D1", "D2", "D3", "D4", "D5")

#id: stable key
#cluster: signals sharing a cluster are expected to correlate
#implementation: beta scales total exposure, xs tilts weights within the universe, exec only improves fills
#build_priority: 1-10 marks how 'likely' a good build is

CATALOGUE = (
    #trend: gradual information diffusion 
    {"id": "trend_absolute_30d", "name": "Absolute momentum, 30d", "family": "trend", "cluster": "TREND",
     "mechanism": "Staged entry by new capital sustains direction.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_absolute_fast", "name": "Absolute momentum, 7-14d", "family": "trend", "cluster": "TREND",
     "mechanism": "Same mechanism, faster decay rate.", "horizon": "2-3d",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_absolute_slow", "name": "Absolute momentum, 90-180d", "family": "trend", "cluster": "TREND",
     "mechanism": "Same mechanism, slowest investor cohort.", "horizon": "1m",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_vol_scaled", "name": "Volatility-scaled trend", "family": "trend", "cluster": "TREND",
     "mechanism": "Trend normalised by realised volatility, removing volatility as a hidden bet.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_breakout", "name": "Donchian breakout", "family": "trend", "cluster": "TREND",
     "mechanism": "Stops and resting orders cluster at range extremes.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_agreement", "name": "Trend agreement across horizons", "family": "trend", "cluster": "TREND",
     "mechanism": "Exposure set by the fraction of lookbacks agreeing; a confidence measure, not a direction.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "trend_agreement_vol", "name": "Trend agreement, volatility scaled", "family": "trend", "cluster": "TREND",
     "mechanism": "Agreement across horizons sets direction and confidence; realised volatility caps each position's risk.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},

    #cross-section: relative pricing within the eligible universe wrt some metric
    {"id": "xs_momentum", "name": "Cross-sectional momentum", "family": "cross_section", "cluster": "XS",
     "mechanism": "Demand shifts sustain relative, not just absolute, trends.", "horizon": "1w",
     "implementation": "xs", "data_tier": "D1", "build_priority": None},
    {"id": "xs_reversal", "name": "Short-term reversal", "family": "cross_section", "cluster": "XS",
     "mechanism": "Compensation for providing liquidity to impatient sellers.", "horizon": "1-3d",
     "implementation": "xs", "data_tier": "D1", "build_priority": None},
    {"id": "xs_residual_momentum", "name": "Residual momentum versus BTC beta", "family": "cross_section", "cluster": "XS_RES",
     "mechanism": "Ranks idiosyncratic return, stripping the market factor that dominates plain cross-sectional momentum.", "horizon": "1w",
     "implementation": "xs", "data_tier": "D1", "build_priority": 3},
    {"id": "xs_size", "name": "Crypto size factor", "family": "cross_section", "cluster": "XS_FAC",
     "mechanism": "Documented size premium in the crypto three-factor evidence.", "horizon": "1m",
     "implementation": "xs", "data_tier": "D1", "build_priority": None},
    {"id": "xs_low_beta", "name": "Betting against beta", "family": "cross_section", "cluster": "XS_FAC",
     "mechanism": "Leverage-constrained buyers overpay for high-beta tokens.", "horizon": "1m",
     "implementation": "xs", "data_tier": "D1", "build_priority": 8},
    {"id": "xs_low_idio_vol", "name": "Low idiosyncratic volatility", "family": "cross_section", "cluster": "XS_FAC",
     "mechanism": "Lottery-preference buyers overpay for high idiosyncratic volatility.", "horizon": "1m",
     "implementation": "xs", "data_tier": "D1", "build_priority": 8},
    {"id": "xs_illiquidity", "name": "Amihud illiquidity premium", "family": "cross_section", "cluster": "XS_FAC",
     "mechanism": "Compensation for bearing price impact.", "horizon": "1m",
     "implementation": "xs", "data_tier": "D1", "build_priority": None},
    {"id": "xs_listing_age", "name": "Listing age and maturity", "family": "cross_section", "cluster": "XS_FAC",
     "mechanism": "Newly listed tokens carry adverse selection and unlock overhang.", "horizon": "1m",
     "implementation": "xs", "data_tier": "D1", "build_priority": None},

    #positioning: derivatives read as a crowding gauge, traded in spot
    {"id": "pos_funding_level", "name": "Funding rate level (z-score)", "family": "positioning", "cluster": "POS",
     "mechanism": "Leveraged longs paying to stay long identifies a fragile configuration.", "horizon": "3d-2w",
     "implementation": "beta", "data_tier": "D3", "build_priority": 2},
    {"id": "pos_funding_rank", "name": "Cross-sectional funding rank", "family": "positioning", "cluster": "POS_XS",
     "mechanism": "Per-token crowding; tilts away from the most crowded names.", "horizon": "1w",
     "implementation": "xs", "data_tier": "D3", "build_priority": None},
    {"id": "pos_funding_dispersion", "name": "Funding dispersion across venues", "family": "positioning", "cluster": "POS",
     "mechanism": "Wide dispersion indicates stressed arbitrage capital.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "pos_oi_price", "name": "Open interest change by price change", "family": "positioning", "cluster": "POS",
     "mechanism": "Rising open interest with rising price is new leveraged longs; falling with rising price is short covering.", "horizon": "1w",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "pos_basis", "name": "Futures basis and term structure", "family": "positioning", "cluster": "POS",
     "mechanism": "Annualised basis is the price of term leverage financing.", "horizon": "2-8w",
     "implementation": "beta", "data_tier": "D3", "build_priority": 7},
    {"id": "vol_put_skew", "name": "25-delta put skew", "family": "positioning", "cluster": "VOL_P",
     "mechanism": "What hedgers are paying for downside insurance.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "vol_iv_rv_spread", "name": "Implied minus realised volatility", "family": "positioning", "cluster": "VOL_P",
     "mechanism": "The variance risk premium; a wide spread means fear is already paid for.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "vol_put_call_oi", "name": "Put/call open interest ratio", "family": "positioning", "cluster": "VOL_P",
     "mechanism": "Cruder positioning gauge; test as a substitute for skew, not an addition.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "micro_liquidation", "name": "Liquidation intensity", "family": "microstructure", "cluster": "MICRO_L",
     "mechanism": "Forced sellers are uninformed, so absorbing them is paid liquidity provision.", "horizon": "1h-3d",
     "implementation": "xs", "data_tier": "D5", "build_priority": None},

    #flow: measurable changes in coin demand or supply
    {"id": "flow_stablecoin_issuance", "name": "Stablecoin net issuance", "family": "flow", "cluster": "FLOW",
     "mechanism": "New fiat entering the system; aggregate demand with no equity analogue.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D4", "build_priority": 10},
    {"id": "flow_etf_creation", "name": "Spot ETF net creation", "family": "flow", "cluster": "FLOW",
     "mechanism": "Creations force real spot buying, on a dated and published schedule.", "horizon": "1-5d",
     "implementation": "beta", "data_tier": "D4", "build_priority": 6},
    {"id": "flow_exchange_outflow", "name": "Exchange net outflow", "family": "flow", "cluster": "FLOW",
     "mechanism": "Coins leaving exchanges reduce sell-side inventory.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D4", "build_priority": None},
    {"id": "event_token_unlock", "name": "Token unlock calendar", "family": "event", "cluster": "EVENT",
     "mechanism": "A scheduled, known-in-advance supply increase; the rare signal with no timing uncertainty.", "horizon": "-7d to +7d",
     "implementation": "xs", "data_tier": "D4", "build_priority": 4},
    {"id": "event_new_listing", "name": "New-listing entry", "family": "event", "cluster": "EVENT",
     "mechanism": "Index-inclusion analogue: mandated and attention-driven buying.", "horizon": "1-10d",
     "implementation": "xs", "data_tier": "D4", "build_priority": None},
    {"id": "flow_address_growth", "name": "Active-address growth", "family": "flow", "cluster": "FLOW_N",
     "mechanism": "Adoption and network usage proxy for fundamental demand.", "horizon": "1-3m",
     "implementation": "xs", "data_tier": "D4", "build_priority": None},
    {"id": "flow_miner_outflow", "name": "Miner outflow and hashprice", "family": "flow", "cluster": "FLOW",
     "mechanism": "Miners are forced sellers with a production cost floor.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D4", "build_priority": None},

    #microstructure: short-horizon, mostly killed by transaction fees
    {"id": "micro_book_imbalance", "name": "Order-book queue imbalance", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Depth asymmetry predicts the next move over seconds to minutes.", "horizon": "1-30m",
     "implementation": "exec", "data_tier": "D2", "build_priority": None},
    {"id": "micro_order_flow_imbalance", "name": "Order-flow imbalance", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Signed trade flow moves price mechanically through the book.", "horizon": "1-30m",
     "implementation": "exec", "data_tier": "D2", "build_priority": None},
    {"id": "micro_venue_dispersion", "name": "Cross-venue price dispersion", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Revolut X quotes may lag deeper venues; a stale-quote measure.", "horizon": "1-15m",
     "implementation": "exec", "data_tier": "D2", "build_priority": None},
    {"id": "micro_btc_lead_lag", "name": "BTC to altcoin lead-lag", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Information impounds in the deepest book first.", "horizon": "1-10m",
     "implementation": "xs", "data_tier": "D2", "build_priority": None},
    {"id": "micro_post_jump", "name": "Post-jump continuation versus reversal", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Jumps with volume are informed and continue; jumps without are flow and revert.", "horizon": "1h-2d",
     "implementation": "xs", "data_tier": "D2", "build_priority": 9},
    {"id": "micro_spread_reversion", "name": "Realised spread mean reversion", "family": "microstructure", "cluster": "MICRO",
     "mechanism": "Spread widening without volume is inventory, not information.", "horizon": "1-60m",
     "implementation": "exec", "data_tier": "D2", "build_priority": None},

    #volatility and risk state: sizing primarily
    {"id": "vol_target", "name": "Realised-volatility targeting", "family": "volatility", "cluster": "VOL",
     "mechanism": "Volatility is autocorrelated where returns are not, so scaling by it stabilises risk.", "horizon": "continuous",
     "implementation": "beta", "data_tier": "D1", "build_priority": 1},
    {"id": "vol_of_vol", "name": "Volatility-of-volatility regime", "family": "volatility", "cluster": "VOL",
     "mechanism": "Unstable volatility precedes regime change; a second-moment signal.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D2", "build_priority": None},
    {"id": "vol_semivariance", "name": "Downside semivariance timing", "family": "volatility", "cluster": "VOL",
     "mechanism": "Asymmetric risk is what actually hurts a long-only book.", "horizon": "1-4w",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "macro_equity_correlation", "name": "Crypto-equity correlation regime", "family": "macro", "cluster": "MACRO",
     "mechanism": "Crypto's diversification value is itself time-varying.", "horizon": "1-3m",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},

    #calendar and macro: flows with clocks attached
    {"id": "cal_execution_window", "name": "Time-of-day execution window", "family": "calendar", "cluster": "CAL",
     "mechanism": "Liquidity is predictably deepest in some windows; Monday 00:05 UTC is Sunday evening in New York.", "horizon": "per trade",
     "implementation": "exec", "data_tier": "D2", "build_priority": 5},
    {"id": "cal_weekend_gap", "name": "Weekend and CME gap", "family": "calendar", "cluster": "CAL",
     "mechanism": "CME closes while spot does not; the gap either fills or does not.", "horizon": "1-3d",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "cal_option_expiry", "name": "Options expiry", "family": "calendar", "cluster": "CAL",
     "mechanism": "Delta-hedge unwind on the last Friday of the month is a dated, mechanical flow.", "horizon": "+/-2d",
     "implementation": "beta", "data_tier": "D3", "build_priority": None},
    {"id": "cal_month_end", "name": "Month and quarter end", "family": "calendar", "cluster": "CAL",
     "mechanism": "Fund rebalancing on a fixed calendar.", "horizon": "+/-2d",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "macro_release_window", "name": "Macro release windows", "family": "macro", "cluster": "MACRO",
     "mechanism": "Crypto's beta to rate expectations became material after 2022.", "horizon": "+/-1d",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},
    {"id": "macro_dollar_yields", "name": "Dollar and real yields", "family": "macro", "cluster": "MACRO",
     "mechanism": "Dollar liquidity is the common factor behind risk assets.", "horizon": "1-3m",
     "implementation": "beta", "data_tier": "D1", "build_priority": None},

    #attention: weakest mechanisms, cheapest data, keep the position small
    {"id": "attn_search_volume", "name": "Search-volume growth", "family": "attention", "cluster": "ATTN",
     "mechanism": "Retail attention precedes retail flow.", "horizon": "1-4w",
     "implementation": "xs", "data_tier": "D4", "build_priority": None},
    {"id": "attn_social_sentiment", "name": "Social and news sentiment", "family": "attention", "cluster": "ATTN",
     "mechanism": "Same mechanism as search volume, faster and noisier.", "horizon": "1-7d",
     "implementation": "xs", "data_tier": "D4", "build_priority": None},
)

_REQUIRED_KEYS = ("id", "name", "family", "cluster", "mechanism", "horizon", "implementation", "data_tier", "build_priority")


def validate_catalogue(entries=CATALOGUE):
    #if malformed catalogue, then its a silent source of double counting risk
    if not entries:
        raise ValueError("The catalogue is empty.")

    seen = set()
    for entry in entries:
        missing = set(_REQUIRED_KEYS).difference(entry)
        if missing:
            raise ValueError(f"Catalogue entry missing keys: {sorted(missing)}")
        if entry["id"] in seen:
            raise ValueError(f"Duplicate catalogue id '{entry['id']}'.")
        seen.add(entry["id"])
        if entry["family"] not in FAMILIES:
            raise ValueError(f"Unknown family '{entry['family']}' for '{entry['id']}'.")
        if entry["implementation"] not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation '{entry['implementation']}' for '{entry['id']}'.")
        if entry["data_tier"] not in DATA_TIERS:
            raise ValueError(f"Unknown data tier '{entry['data_tier']}' for '{entry['id']}'.")
        priority = entry["build_priority"]
        if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 10):
            raise ValueError(f"build_priority must be None or 1-10 for '{entry['id']}'.")

    return entries


def catalogue_frame():
    #join registration status onto the catalogue as a frame
    from .registry import available

    validate_catalogue()
    frame = pd.DataFrame(list(CATALOGUE))
    registered = set(available())
    frame["status"] = frame["id"].map(lambda key: "registered" if key in registered else "planned")

    return frame.set_index("id").sort_index()


def entry(signal_id):
    #one catalogue entry by id
    for record in CATALOGUE:
        if record["id"] == signal_id:
            return dict(record)

    raise ValueError(f"'{signal_id}' is not in the catalogue.")


def cluster_of(signal_id):
    #the cluster a signal is expected to sit in

    return entry(signal_id)["cluster"]


def build_queue():
    #order all signals by order of importance
    frame = catalogue_frame()
    queued = frame.loc[frame["build_priority"].notna()].copy()

    return queued.sort_values(["build_priority", "name"])


validate_catalogue()
