# stats_fetcher.py (Final version with advanced debugging)
import pandas as pd
import requests
import os
import difflib

LEAGUE_CSV_URLS = {
    "E0.csv": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "SP1.csv": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "I1.csv": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "D1.csv": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "F1.csv": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
}

def _normalize_team_name(name: str) -> str:
    """
    Normalizes team names and uses an expanded translation dictionary.
    """
    name = name.lower().replace(" fc", "").replace(" afc", "").replace(" cf", "").replace(" & ", " and ")
    
    team_name_map = {
        # English Premier League
        "wolverhampton wanderers": "wolves",
        "nottingham forest": "nott'm forest",
        "manchester city": "man city",
        "manchester united": "man utd",
        "newcastle united": "newcastle",
        "tottenham hotspur": "tottenham",
        "brighton and hove albion": "brighton",
        "west ham united": "west ham",
        # Spanish La Liga
        "atlético madrid": "atletico madrid",
        "real betis": "betis",
        "cádiz": "cadiz",
        "deportivo alavés": "alaves",
        "atlético de madrid": "atletico madrid",
        "sevilla": "sevilla", # Ensures exact match
    }
    return team_name_map.get(name, name)

def load_stats_data_from_csv(filepath: str):
    """
    Loads statistics data from a CSV file, downloading it automatically if it doesn't exist.
    """
    if not os.path.exists(filepath):
        print(f"⚠️ File '{filepath}' not found. Attempting to download...")
        filename = os.path.basename(filepath)
        url = LEAGUE_CSV_URLS.get(filename)
        if not url:
            print(f"❌ No known download URL for '{filename}'.")
            return None
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ Successfully downloaded '{filepath}'.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download file: {e}")
            return None

    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1', on_bad_lines='skip')
        rename_map = {
            'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'FTR': 'Result',
            'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
            'xG': 'Home_xG', 'xG.1': 'Away_xG', 'xGA': 'Away_xG_for_Home', 'xGA.1': 'Home_xG_for_Away'
        }
        df.rename(columns=lambda c: rename_map.get(c, c), inplace=True)
        
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
        if not all(col in df.columns for col in required_cols):
            return None
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading stats CSV: {e}")
        return None

def find_team_stats(team_name_to_find: str, league_df: pd.DataFrame, debug=True):
    """
    A smart search function that uses fuzzy matching and provides a detailed debug report.
    """
    if league_df is None or league_df.empty: return None

    best_match_score = 0.6
    best_match_stats = None
    
    name_to_find_norm = _normalize_team_name(team_name_to_find)
    
    # Create a list of all normalized names from the DataFrame for comparison
    available_team_names = {
        _normalize_team_name(name): name for name in league_df['HomeTeam'].unique()
    }

    # Find the best match using difflib
    closest_matches = difflib.get_close_matches(name_to_find_norm, available_team_names.keys(), n=5, cutoff=0.5)

    if debug:
        print("\n" + "="*50)
        print(f"🔍 DEBUG: Searching for '{team_name_to_find}' (Normalized to '{name_to_find_norm}')")
        print("---")
        if closest_matches:
            print("Top 5 closest matches found in the statistics file:")
            for match in closest_matches:
                score = difflib.SequenceMatcher(None, name_to_find_norm, match).ratio()
                print(f"  - Match: '{match}' (Original: '{available_team_names[match]}') | Similarity Score: {score:.2f}")
        else:
            print("❌ No close matches found at all.")
        print("---")

    if closest_matches:
        best_match_normalized = closest_matches[0]
        final_score = difflib.SequenceMatcher(None, name_to_find_norm, best_match_normalized).ratio()
        
        if final_score >= best_match_score:
            original_best_match_name = available_team_names[best_match_normalized]
            row = league_df[league_df['HomeTeam'] == original_best_match_name].iloc[0] # Use any row for the team to get stats
            
            # Perform calculations
            league_total_goals = league_df['HomeGoals'].sum() + league_df['AwayGoals'].sum()
            total_games = len(league_df)
            avg_goals = league_total_goals / total_games

            team_games = league_df[(league_df['HomeTeam'] == original_best_match_name) | (league_df['AwayTeam'] == original_best_match_name)]
            attack_strength = (team_games['HomeGoals'].sum() + team_games['AwayGoals'].sum()) / len(team_games) / (avg_goals / 2)
            defense_strength = (team_games['HomeGoals'].sum() + team_games['AwayGoals'].sum()) / len(team_games) / (avg_goals / 2)

            best_match_stats = {
                "attack": attack_strength, "defense": defense_strength,
                "found_name": original_best_match_name, "score": final_score
            }
            if debug: print(f"✅ DEBUG: Match found! '{original_best_match_name}' with score {final_score:.2f}")
    
    if debug and not best_match_stats:
        print(f"❌ DEBUG: No match found with score >= {best_match_score}.")
    
    if debug: print("="*50 + "\n")

    return best_match_stats
