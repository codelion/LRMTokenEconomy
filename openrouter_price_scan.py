#!/usr/bin/env python3
"""openrouter_price_scan.py

Extract pricing statistics for a list of OpenRouter models.

Given a JSON configuration file produced by your own LLM‑runner
(the same structure as the provided `query_config_full.json`), the script
hits OpenRouter's public REST API and gathers provider‑level pricing for
all referenced models.  It then writes a CSV with the following columns:

    model_name,              # Uses 'name' field from config (not model ID)
    provider_count,
    prompt_min,           # $ per 1M tokens
    prompt_max,           # $ per 1M tokens
    prompt_avg,           # $ per 1M tokens
    completion_min,       # $ per 1M tokens
    completion_max,       # $ per 1M tokens
    completion_avg,       # $ per 1M tokens
    max_output_tokens_min,
    max_output_tokens_max,
    max_output_tokens_avg,

* **max_output_tokens_min/max/avg** – statistics computed from all providers'
  max_completion_tokens values for this model.

Note: Throughput data (tokens per second) is not available through the OpenRouter API.
(the same structure as the provided `query_config_full.json`), the script
hits OpenRouter’s public REST API and gathers provider‑level pricing for
all referenced models.  It then writes a CSV with the following columns:

    model_name,
    provider_count,
    prompt_min,
    prompt_max,
    prompt_avg,
    completion_min,
    completion_max,
    completion_avg,
    throughput,
    max_output_tokens

* **throughput** – currently always "N/A"; OpenRouter does not expose a
  numeric throughput metric yet.
* **max_output_tokens** – taken from the model’s `top_provider` entry
  inside `/api/v1/models`.

Usage
-----
```bash
python openrouter_price_scan.py \
    --config query_config_full.json \
    --out model_prices.csv
```

Notes
-----
* The script is intentionally synchronous (no asyncio) for simplicity.
* Requires an optional `OPENROUTER_API_KEY` env variable – requests work
  without it for now, but supplying it future‑proofs against rate‑limits.
* Any model that cannot be found (404) is reported to `stderr` and
  skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from statistics import mean
from typing import Dict, List, Tuple

import requests

API_BASE = "https://openrouter.ai/api/v1"
CSV_HEADER = [
    "model_name",          # Uses 'name' field from config
    "provider_count",
    "prompt_min",
    "prompt_max",
    "prompt_avg",
    "completion_min",
    "completion_max",
    "completion_avg",
    "max_output_tokens_min",
    "max_output_tokens_max",
    "max_output_tokens_avg",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_headers() -> Dict[str, str]:
    """Return request headers with optional Authorization."""
    headers = {"User-Agent": "openrouter-price-scan/1.0"}
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def strip_variant(model_id: str) -> Tuple[str, str]:
    """Split `author/slug(:variant)` → (author, slug).

    If the string is malformed, ValueError is raised.
    """
    if "/" not in model_id:
        raise ValueError(f"Unexpected model id format: {model_id!r}")
    author, rest = model_id.split("/", 1)
    slug = rest.split(":", 1)[0]  # discard variant suffix
    return author, slug


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def fetch_endpoints(author: str, slug: str) -> List[dict] | None:
    """GET /models/{author}/{slug}/endpoints.

    Returns list of endpoints or None if 404/not found.
    """
    url = f"{API_BASE}/models/{author}/{slug}/endpoints"
    res = requests.get(url, headers=get_headers(), timeout=30)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    # Historical API responses used either top‑level "endpoints" _or_
    # nested {"data": {"endpoints": [...]}} – support both.
    body = res.json()
    if "endpoints" in body:
        return body["endpoints"]
    return body.get("data", {}).get("endpoints", [])


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def decimal_mean(values: List[Decimal]) -> Decimal:
    return (sum(values) / len(values)).quantize(Decimal("0.0000001"),
                                                rounding=ROUND_HALF_UP)


def summarise_prices(endpoints: List[dict]) -> Tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Return (p_min, p_max, p_avg, c_min, c_max, c_avg)."""
    prompt_prices: List[Decimal] = []
    completion_prices: List[Decimal] = []
    for ep in endpoints:
        pricing = ep.get("pricing", {})
        try:
            if "prompt" in pricing:
                prompt_prices.append(Decimal(str(pricing["prompt"])) )
            if "completion" in pricing:
                completion_prices.append(Decimal(str(pricing["completion"])) )
        except InvalidOperation:
            continue  # skip non‑numeric pricing values
    if not prompt_prices or not completion_prices:
        raise ValueError("No valid pricing found in endpoints data")

    p_min = min(prompt_prices)
    p_max = max(prompt_prices)
    p_avg = decimal_mean(prompt_prices)
    c_min = min(completion_prices)
    c_max = max(completion_prices)
    c_avg = decimal_mean(completion_prices)
    return p_min, p_max, p_avg, c_min, c_max, c_avg


def summarise_max_output_tokens(endpoints: List[dict]) -> Tuple[int, int, int]:
    """Return (min, max, avg) for max_completion_tokens from all endpoints."""
    max_tokens_values = []
    for ep in endpoints:
        max_tokens = ep.get("max_completion_tokens")
        if max_tokens is not None and isinstance(max_tokens, (int, float)) and max_tokens > 0:
            max_tokens_values.append(int(max_tokens))
    
    if not max_tokens_values:
        return 0, 0, 0
    
    return min(max_tokens_values), max(max_tokens_values), int(sum(max_tokens_values) / len(max_tokens_values))


def process_models(models_to_query: List[Tuple[str, str]], out_path: str) -> None:
    """Main loop: fetch, compute, and write CSV.
    
    Args:
        models_to_query: List of (name, model_id) tuples
        out_path: Output CSV file path
    """
    rows = []
    for name, model_id in models_to_query:
        try:
            author, slug = strip_variant(model_id)
        except ValueError as exc:
            print(f"Skipping malformed id {model_id!r}: {exc}", file=sys.stderr)
            continue

        endpoints = fetch_endpoints(author, slug)
        if endpoints is None or not endpoints:
            print(f"Model not found on OpenRouter → skipped: {model_id}", file=sys.stderr)
            continue

        # Pricing statistics
        try:
            p_min, p_max, p_avg, c_min, c_max, c_avg = summarise_prices(endpoints)
        except ValueError as err:
            print(f"No pricing data for {model_id}: {err}", file=sys.stderr)
            continue

        # Max output tokens from endpoints
        mo_min, mo_max, mo_avg = summarise_max_output_tokens(endpoints)

        # Convert to $ per 1M tokens
        p_min_per_1m = p_min * 1_000_000
        p_max_per_1m = p_max * 1_000_000
        p_avg_per_1m = p_avg * 1_000_000
        c_min_per_1m = c_min * 1_000_000
        c_max_per_1m = c_max * 1_000_000
        c_avg_per_1m = c_avg * 1_000_000

        rows.append([
            name,
            len(endpoints),
            f"{p_min_per_1m:.2f}", f"{p_max_per_1m:.2f}", f"{p_avg_per_1m:.2f}",
            f"{c_min_per_1m:.2f}", f"{c_max_per_1m:.2f}", f"{c_avg_per_1m:.2f}",
            mo_min, mo_max, mo_avg,
        ])

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} models → {out_path}")


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenRouter pricing stats from a config file.")
    parser.add_argument("--config", default="query_config_full.json", help="Path to JSON config (default: %(default)s)")
    parser.add_argument("--out", default="model_prices.csv", help="Output CSV path (default: %(default)s)")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        sys.exit(f"❌ Cannot read config: {exc}")

    # Extract (name, model) pairs from config
    model_entries = []
    for entry in cfg.get("llms", []):
        if entry.get("model") and entry.get("name"):
            model_entries.append((entry["name"], entry["model"]))
    
    if not model_entries:
        sys.exit("❌ No entries with both 'name' and 'model' found in the config file.")

    # Remove duplicates while preserving order (based on model ID)
    seen = set()
    unique_models = []
    for name, model_id in model_entries:
        if model_id not in seen:
            unique_models.append((name, model_id))
            seen.add(model_id)

    process_models(unique_models, args.out)


if __name__ == "__main__":
    main()
