import pandas as pd

POLICY_VERSION = "meme_exclusions_v1" #meme coins are excluded by my personal opinion.
POLICY_REVIEWED_AT = "2026-09-18"

#research scope chosen now and applied retrospectively.
MEME_EXCLUSIONS = {
    "DOGE-USD": ("Doge meme origins", "https://dogecoin.com/dogepedia/articles/history-of-dogecoin/"),
    "SHIB-USD": ("Meme-inspired origins", "https://www.coinbase.com/price/shiba-inu"),
    "BONK-USD": ("Dog-themed internet-culture community token", "https://www.bonkcoin.com/about"),
    "FLOKI-USD": ("Meme origins, despite subsequent utility", "https://floki.com/"),
    "PEPE-USD": ("Pepe the Frog meme token", "https://www.coinbase.com/en-gb/price/pepe"),
    "WIF-USD": ("Dogwifhat memecoin", "https://www.coinbase.com/converter/bonk/wif"),
    "SPX-USD": ("Stock-market-themed meme coin", "https://www.spx6900.com/"),
    "TRUMP-USD": ("Explicitly marketed as a Trump meme token", "https://gettrumpmemes.com/"),
    "PENGU-USD": ("Brand-linked token with explicit memecoin positioning", "https://www.pudgypenguins.com/blogs/news/" "pudgy-penguins-featured-on-f1-singapore-rear-wing"),
}


def meme_exclusion_table():
    #return versioned exclusions
    table = pd.DataFrame.from_dict(MEME_EXCLUSIONS, orient="index", columns=["exclusion_reason", "source_url"]).rename_axis("product_id")

    table["policy_version"] = POLICY_VERSION
    table["reviewed_at"] = POLICY_REVIEWED_AT
    return table.sort_index()


def apply_universe_policy(eligibility_panel):
    #add scope eligibility without deleting prices or prior flags

    required = {"product_id", "research_candidate"}
    missing = required.difference(eligibility_panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    candidates = eligibility_panel["research_candidate"]
    if (not pd.api.types.is_bool_dtype(candidates) or candidates.isna().any()):
        raise ValueError("research_candidate must contain non-null booleans.")

    frame = eligibility_panel.copy()
    exclusions = meme_exclusion_table()

    frame["meme_excluded"] = frame["product_id"].isin(exclusions.index)
    frame["scope_exclusion_reason"] = frame["product_id"].map(exclusions["exclusion_reason"]).fillna("")
    frame["universe_candidate"] = (frame["research_candidate"] & ~frame["meme_excluded"])
    frame["universe_policy_version"] = POLICY_VERSION
    return frame