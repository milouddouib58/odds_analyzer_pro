# stats_fetcher.py (النسخة المصححة مع مترجم الأسماء)
import os
import requests
import pandas as pd

API_URL = "https://api.football-data.org/v4/"

def _normalize_team_name(name: str) -> str:
    """
    يوحد أسماء الفرق لتسهيل العثور عليها بين مصادر البيانات المختلفة.
    مثال: "Manchester United FC" -> "manchester united"
    """
    # يمكنك إضافة استثناءات خاصة هنا إذا لزم الأمر
    # special_cases = {"Wolverhampton Wanderers": "Wolves"}
    # if name in special_cases:
    #     name = special_cases[name]
    
    return name.lower().replace(" fc", "").replace(" afc", "").replace(" & ", " and ")

def get_league_stats_from_api(api_key: str, competition_code: str = "PL"):
    """
    يجلب إحصائيات الدوري باستخدام مفتاح football-data.org الرسمي.
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
            print("❌ هيكل البيانات غير متوقع من football-data.org")
            return None

        df = pd.DataFrame(standings[0]['table'])
        df['team_name'] = df['team'].apply(lambda x: x['name'])

        league_total_goals = df['goalsFor'].sum()
        league_total_matches = df['playedGames'].sum() / 2
        
        # تجنب القسمة على صفر إذا لم تكن هناك مباريات
        if league_total_matches == 0: return None
        avg_goals_per_match = league_total_goals / league_total_matches

        df['attack_strength'] = (df['goalsFor'] / df['playedGames']) / (avg_goals_per_match / 2)
        df['defense_strength'] = (df['goalsAgainst'] / df['playedGames']) / (avg_goals_per_match / 2)

        team_stats = {}
        for index, row in df.iterrows():
            normalized_name = _normalize_team_name(row['team_name'])
            team_stats[normalized_name] = {
                "attack": row['attack_strength'],
                "defense": row['defense_strength']
            }
        
        print("✅ تم سحب الإحصائيات بنجاح من football-data.org!")
        return team_stats

    except Exception as e:
        print(f"❌ حدث خطأ أثناء سحب الإحصائيات من API: {e}")
        return None
