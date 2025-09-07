# stats_fetcher.py (النسخة المطورة بـ football-data API)
import os
import requests
import pandas as pd

API_URL = "https://api.football-data.org/v4/"

def get_league_stats_from_api(api_key: str, competition_code: str = "PL"):
    """
    يجلب إحصائيات الدوري باستخدام مفتاح football-data.org الرسمي.
    PL = Premier League, BL1 = Bundesliga, SA = Serie A, etc.
    """
    if not api_key:
        raise ValueError("مفتاح football-data.org API مطلوب.")

    headers = {"X-Auth-Token": api_key}
    url = f"{API_URL}competitions/{competition_code}/standings"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # يتأكد إذا كان الطلب ناجحًا
        data = response.json()

        # استخراج جدول الترتيب
        standings = data['standings'][0]['table']
        df = pd.DataFrame(standings)

        # استخراج بيانات الفرق الأساسية
        # "position", "team", "playedGames", "won", "draw", "lost", "points", "goalsFor", "goalsAgainst", "goalDifference"
        df['team_name'] = df['team'].apply(lambda x: x['name'])

        # --- حساب المتوسطات ---
        league_total_goals = df['goalsFor'].sum()
        league_total_matches = df['playedGames'].sum() / 2
        avg_goals_per_match = league_total_goals / league_total_matches

        # --- حساب قوة الهجوم والدفاع ---
        df['attack_strength'] = (df['goalsFor'] / df['playedGames']) / (avg_goals_per_match / 2)
        df['defense_strength'] = (df['goalsAgainst'] / df['playedGames']) / (avg_goals_per_match / 2)

        # تجهيز القاموس النهائي بالنتائج
        team_stats = {}
        for index, row in df.iterrows():
            # قد تحتاج لتعديل أسماء الفرق لتتوافق مع أسماء odds api
            # مثال: "Manchester United FC" vs "Manchester United"
            team_name = row['team_name'].replace(" FC", "").replace(" AFC", "")
            
            team_stats[team_name] = {
                "attack": row['attack_strength'],
                "defense": row['defense_strength']
            }
        
        print("✅ تم سحب الإحصائيات بنجاح من football-data.org!")
        return team_stats

    except Exception as e:
        print(f"❌ حدث خطأ أثناء سحب الإحصائيات من API: {e}")
        error_content = response.json()
        print("رسالة الخطأ من السيرفر:", error_content.get('message'))
        return None

# # للتجربة
# if __name__ == '__main__':
#     # استبدل "YOUR_API_KEY" بمفتاحك
#     api_key = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_API_KEY") 
#     stats = get_league_stats_from_api(api_key=api_key)
#     if stats:
#         print(stats.get("Arsenal"))
