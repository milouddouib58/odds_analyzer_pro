# stats_fetcher.py (النسخة المصححة من الخطأ المطبعي)
import os
import requests
import pandas as pd
import difflib

API_URL = "https://api.football-data.org/v4/"

def _normalize_team_name(name: str) -> str:
    """
    يوحد أسماء الفرق لتسهيل العثور عليها.
    """
    name = name.lower().replace(" fc", "").replace(" afc", "").replace(" & ", " and ")
    
    team_name_map = {
        "wolverhampton wanderers": "wolves",
        "nottingham forest": "nott'm forest",
        "manchester city": "man city",
        "manchester united": "man utd",
        "newcastle united": "newcastle",
        "tottenham hotspur": "tottenham",
        "brighton and hove albion": "brighton",
        "west ham united": "west ham",
        "leicester city": "leicester",
        "afc bournemouth": "bournemouth",
    }
    
    return team_name_map.get(name, name)

def get_league_stats_from_api(api_key: str, competition_code: str = "PL"):
    """
    يجلب جدول الترتيب الكامل من football-data.org.
    """
    if not api_key:
        raise ValueError("مفتاح football-data.org API مطلوب.")

    headers = {"X-Auth-Token": api_key}
    url = f"{API_URL}competitions/{competition_code}/standings"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        standings = data.get('standings', [])
        if not standings or 'table' not in standings[0]:
            return None
            
        df = pd.DataFrame(standings[0]['table'])
        df['original_team_name'] = df['team'].apply(lambda x: x.get('name', ''))
        
        print("✅ تم سحب جدول الإحصائيات بنجاح من football-data.org!")
        return df

    except Exception as e:
        print(f"❌ حدث خطأ أثناء سحب الإحصائيات من API: {e}")
        return None

def find_team_stats(team_name_to_find: str, league_df: pd.DataFrame, debug=True):
    """
    دالة بحث ذكية ومرنة مع وضع التصحيح (Debug Mode) لإظهار عملية البحث.
    """
    if league_df is None or league_df.empty:
        return None

    best_match_score = 0.6
    best_match_stats = None

    name_to_find_norm = _normalize_team_name(team_name_to_find)

    if debug:
        print("\n" + "="*50)
        print(f"🔍 DEBUG: البحث عن الفريق '{team_name_to_find}' (تم توحيده إلى '{name_to_find_norm}')")
        print("---")

    all_team_names_from_df = league_df['original_team_name'].tolist()

    for index, row in league_df.iterrows():
        original_name = row['original_team_name']
        normalized_name_from_df = _normalize_team_name(original_name)
        
        score = difflib.SequenceMatcher(None, name_to_find_norm, normalized_name_from_df).ratio()
        
        if debug:
            if score > 0.3:
                print(f"  - مقارنة مع '{normalized_name_from_df}': درجة التشابه = {score:.2f}")

        if score > best_match_score:
            best_match_score = score
            
            league_total_goals = league_df['goalsFor'].sum()
            league_total_matches = league_df['playedGames'].sum() / 2
            avg_goals_per_match = league_total_goals / league_total_matches

            attack_strength = (row['goalsFor'] / row['playedGames']) / (avg_goals_per_match / 2)
            # --- ::: This is the corrected line ::: ---
            defense_strength = (row['goalsAgainst'] / row['playedGames']) / (avg_goals_per_match / 2)
            
            best_match_stats = {
                "attack": attack_strength,
                "defense": defense_strength,
                "found_name": original_name,
                "score": best_match_score
            }
    
    if debug:
        if best_match_stats:
            print(f"✅ DEBUG: أفضل تطابق تم العثور عليه هو '{best_match_stats['found_name']}' بدرجة {best_match_stats['score']:.2f}")
        else:
            print(f"❌ DEBUG: لم يتم العثور على تطابق كافٍ. أعلى درجة تشابه كانت أقل من {best_match_score}")
            print("--- قائمة الأسماء المتوفرة في الإحصائيات ---")
            for name in all_team_names_from_df:
                print(f"  -> {_normalize_team_name(name)}")
        print("="*50 + "\n")

    return best_match_stats
