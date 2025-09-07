# odds_math.py
import math
import pandas as pd

# ... (كل الدوال القديمة مثل aggregate_prices, shin_fair_probs, etc. تبقى كما هي) ...

def calculate_form_probs(team_name: str, opponent_name: str, stats_df: pd.DataFrame, num_matches=6):
    """
    يحسب احتمالات الفوز بناءً على أداء الفريقين في آخر N مباريات.
    """
    team_matches = stats_df[(stats_df['HomeTeam'] == team_name) | (stats_df['AwayTeam'] == team_name)].tail(num_matches)
    opponent_matches = stats_df[(stats_df['HomeTeam'] == opponent_name) | (stats_df['AwayTeam'] == opponent_name)].tail(num_matches)

    if team_matches.empty or opponent_matches.empty:
        return {"home": 0.33, "draw": 0.33, "away": 0.33} # نتيجة محايدة

    def get_ppg(df, team):
        points = 0
        for _, row in df.iterrows():
            if row['HomeTeam'] == team and row['Result'] == 'H': points += 3
            elif row['AwayTeam'] == team and row['Result'] == 'A': points += 3
            elif row['Result'] == 'D': points += 1
        return points / len(df)

    team_ppg = get_ppg(team_matches, team_name)
    opponent_ppg = get_ppg(opponent_matches, opponent_name)
    
    total_ppg = team_ppg + opponent_ppg
    if total_ppg == 0: return {"home": 0.33, "draw": 0.33, "away": 0.33}

    # تقدير بسيط للاحتمالات بناءً على فارق النقاط
    home_prob = team_ppg / total_ppg
    away_prob = opponent_ppg / total_ppg
    # نفترض أن التعادل يأخذ نسبة ثابتة من الفروقات
    draw_prob = 1 - abs(home_prob - away_prob) * 0.5
    
    home_prob *= (1 - draw_prob)
    away_prob *= (1 - draw_prob)

    total_final = home_prob + draw_prob + away_prob
    return {"home": home_prob/total_final, "draw": draw_prob/total_final, "away": away_prob/total_final}


def calculate_xg_probs(home_team: str, away_team: str, stats_df: pd.DataFrame):
    """
    يحسب احتمالات الفوز باستخدام متوسط الأهداف المتوقعة (xG).
    """
    home_data = stats_df[stats_df['HomeTeam'] == home_team]
    away_data = stats_df[stats_df['AwayTeam'] == away_team]

    if home_data.empty or away_data.empty:
        return None

    # قوة هجوم ودفاع كل فريق بناءً على xG
    home_attack_xg = home_data['Home_xG'].mean()
    home_defense_xg = home_data['Home_xG_for_Away'].mean()
    away_attack_xg = away_data['Away_xG'].mean()
    away_defense_xg = away_data['Away_xG_for_Home'].mean()

    # استخدام نموذج بواسون لكن بمعطيات xG
    return poisson_prediction(home_attack_xg, home_defense_xg, away_attack_xg, away_defense_xg)
