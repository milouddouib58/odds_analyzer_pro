# odds_provider_theoddsapi.py
import os
import requests

API_URL = "https://api.the-odds-api.com/v4"
DEFAULT_TIMEOUT = 12

def _get_api_key():
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set.")
    return api_key

def list_sports():
    url = f"{API_URL}/sports/"
    params = {"apiKey": _get_api_key()}
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()

def fetch_odds_for_sport(sport_key: str, regions: str, markets: str):
    url = f"{API_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": _get_api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json(), response.headers

def extract_h2h_prices(event: dict) -> dict:
    prices = {"home": [], "draw": [], "away": []}
    home_team = event.get("home_team")
    away_team = event.get("away_team")
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").strip()
                    price = outcome.get("price")
                    if not price:
                        continue
                    if name == home_team:
                        prices["home"].append(price)
                    elif name == away_team:
                        prices["away"].append(price)
                    else:
                        # دعم draw أو tie
                        if name.lower() in ("draw", "tie"):
                            prices["draw"].append(price)
    return prices

def extract_totals_lines(event: dict) -> dict:
    lines = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "totals":
                for outcome in market.get("outcomes", []):
                    line_point = outcome.get("point")
                    if line_point is None:
                        continue  # تجاهل النقاط غير المحددة
                    price = outcome.get("price")
                    side = (outcome.get("name") or "").strip().lower()
                    if line_point not in lines:
                        lines[line_point] = {"over": [], "under": []}
                    if side in lines[line_point] and price:
                        lines[line_point][side].append(price)
    # إعادة كمفاتيح نصية للعرض
    return {str(k): v for k, v in lines.items()}
