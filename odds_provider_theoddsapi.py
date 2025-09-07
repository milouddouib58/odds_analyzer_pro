# odds_provider_theoddsapi.py
import os
import requests

API_URL = "https://api.the-odds-api.com/v4"

def _get_api_key():
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set.")
    return api_key

def list_sports():
    url = f"{API_URL}/sports/?apiKey={_get_api_key()}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_odds_for_sport(sport_key: str, regions: str, markets: str):
    url = f"{API_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": _get_api_key(),
        "regions": regions,
        "markets": markets,
    }
    response = requests.get(url, params=params)
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
                    price = outcome.get("price")
                    if outcome.get("name") == home_team:
                        prices["home"].append(price)
                    elif outcome.get("name") == away_team:
                        prices["away"].append(price)
                    elif outcome.get("name") == "Draw":
                        prices["draw"].append(price)
    return prices

def extract_totals_lines(event: dict) -> dict:
    lines = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "totals":
                for outcome in market.get("outcomes", []):
                    line_point = outcome.get("point")
                    price = outcome.get("price")
                    side = outcome.get("name").lower() # "Over" or "Under"
                    
                    if line_point not in lines:
                        lines[line_point] = {"over": [], "under": []}
                    
                    if side in lines[line_point]:
                        lines[line_point][side].append(price)
    return {str(k): v for k, v in lines.items()}
