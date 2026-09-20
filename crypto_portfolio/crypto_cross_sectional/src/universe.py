import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_gbp_inventory(snapshot_dir, scope_exclusions):
    #keep every GBP pair. record checks and scope exclusions

    saved = {}
    for name in ("pairs", "currencies", "tickers"):
        record = json.loads((Path(snapshot_dir) / f"{name}.json").read_text(encoding="utf-8"))
        if record["requested_region"] != "UK" or record["http_status"] != 200:
            raise ValueError(f"Unexpected region or response status: {name}")
        saved[name] = record["payload"]

    pairs = pd.DataFrame.from_dict(saved["pairs"], orient="index")
    inventory = pairs.loc[pairs["quote"].eq("GBP")].copy()
    inventory.index.name = "symbol"
    if inventory.empty:
        raise ValueError("No GBP pairs found.")

    currencies = pd.DataFrame.from_dict(saved["currencies"], orient="index")
    for field in ("name", "asset_type", "status"):
        inventory[f"currency_{field}"] = inventory["base"].map(currencies[field])

    tickers = pd.DataFrame(saved["tickers"]["data"]).set_index("symbol")
    if not tickers.index.is_unique or not tickers["region"].eq("UK").all():
        raise ValueError("Duplicate symbols or non-UK quotes.")

    inventory = inventory.join(tickers[["bid", "ask", "quote_volume_24h"]], validate="one_to_one")
    numeric = ["bid", "ask", "quote_volume_24h"]
    inventory[numeric] = inventory[numeric].apply(pd.to_numeric, errors="coerce")

    quote_valid = (np.isfinite(inventory[["bid", "ask"]]).all(axis=1) & inventory["bid"].gt(0) & inventory["ask"].ge(inventory["bid"]))
    volume_valid = (np.isfinite(inventory["quote_volume_24h"]) & inventory["quote_volume_24h"].ge(0))
    mid = (inventory["bid"] + inventory["ask"]) / 2
    inventory["spread_bps"] = (10_000 * (inventory["ask"] - inventory["bid"]) / mid.where(quote_valid))
    inventory["scope_exclusion"] = (inventory["base"].map(scope_exclusions).fillna(""))

    failures = pd.DataFrame({
        "pair_not_active": ~inventory["status"].eq("active"),
        "currency_not_active": ~inventory["currency_status"].eq("active"),
        "currency_not_crypto": ~inventory["currency_asset_type"].eq("crypto"),
        "missing_currency_name": inventory["currency_name"].isna(),
        "missing_or_invalid_quote": ~quote_valid,
        "missing_or_invalid_volume": ~volume_valid
    })
    inventory["snapshot_issues"] = failures.apply(lambda row: "; ".join(row.index[row]), axis=1)
    inventory["passes_initial_screen"] = (inventory["snapshot_issues"].eq("") & inventory["scope_exclusion"].eq(""))

    return inventory.sort_values("quote_volume_24h", ascending=False)

def map_coinbase_products(inventory, products):
    #map candidate assets to exact GBP and USD catalogue matches

    catalogue = pd.DataFrame(products)
    required = {"id", "base_currency", "quote_currency", "status"}

    if not required.issubset(catalogue.columns):
        raise ValueError("Missing required Coinbase product fields.")
    if catalogue[list(required)].isna().any().any():
        raise ValueError("Missing values in Coinbase product identifiers/status.")
    if catalogue["id"].duplicated().any():
        raise ValueError("Duplicate Coinbase product IDs.")

    mapping = inventory.loc[inventory["passes_initial_screen"], ["base", "currency_name"]].copy()
    for quote in ("GBP", "USD"):
        matches = catalogue.loc[catalogue["quote_currency"].eq(quote)].set_index("base_currency")
        if not matches.index.is_unique:
            raise ValueError(f"Multiple {quote} products for the same base.")

        prefix = quote.lower()
        mapping[f"{prefix}_product"] = mapping["base"].map(matches["id"])
        mapping[f"{prefix}_status"] = mapping["base"].map(matches["status"])
    mapping["no_catalogue_match"] = (mapping["gbp_product"].isna() & mapping["usd_product"].isna())

    return mapping.sort_index()