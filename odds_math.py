# odds_math.py
import math
import pandas as pd

def aggregate_prices(prices: list, mode: str = 'best') -> float:
    """
    تجميع قائمة من الأسعار. 'best' ترجع أعلى سعر.
    """
    if not prices: return 0.0
    if mode == 'best':
        return max(prices)
    elif mode == 'median':
        prices.sort()
        mid = len(prices) // 2
        return (prices[mid] + prices[~mid]) / 2
    elif mode == 'mean':
        return sum(prices) / len(prices)
    return 0.0

def implied_from_decimal(odds: dict) -> dict:
    """
    تحويل الأسعار العشرية إلى احتمالات ضمنية.
    """
    return {k: 1/v if v and v > 0 else 0 for k, v in odds.items()}

def overround(implied_probs: dict) -> float:
    """
    حساب هامش الربح الإجمالي للسوق (Overround).
    """
    return sum(implied_probs.values())

def normalize_proportional(implied_probs: dict) -> dict:
    """
    تسوية الاحتمالات لإزالة الهامش بالطريقة التناسبية.
    """
    total_prob = overround(implied_probs)
    if total_prob == 0: return implied_probs
    return {k: v / total_prob for k, v in implied_probs.items()}

def shin_fair_probs(implied_probs: dict) -> dict:
    """
    إزالة هامش الربح باستخدام طريقة Shin الأكثر دقة.
    """
    if any(p <= 0 for p in implied_probs.values()):
        return normalize_proportional(implied_probs)
    if not implied_probs: return {}
    
    def sum_diff_sq(z, probs):
        return sum(((p - z) / (1 - z if p < z else p))**2 for p in probs) - 1

    implied_values = list(implied_probs.values())
    total_prob = sum(implied_values)
    if total_prob < 1.0: return normalize_proportional(implied_probs)

    low = 0.0
    high = min(implied_values)
    z = 0.0
    
    for _ in range(100):
        z = (low + high) / 2
        if sum_diff_sq(z, implied_values) > 0:
            low = z
        else:
            high = z
            
    fair_probs_values = [(p - z) / (1 - z) for p in implied_values]
    return dict(zip(implied_probs.keys(), fair_probs_values))

def kelly_suggestions(fair_probs: dict, book_odds: dict, bankroll: float, kelly_scale: float = 0.25) -> dict:
    """
    حساب اقتراحات الرهان بناءً على معيار كيلي.
    """
    suggestions = {}
    for outcome, prob in fair_probs.items():
        odds = book_odds.get(outcome)
        if not odds or odds <= 1: continue
        
        edge = (prob * odds) - 1
        
        if edge > 0:
            kelly_fraction = edge / (odds - 1)
            stake_fraction = kelly_fraction * kelly_scale
            stake_amount = bankroll * stake_fraction
            
            suggestions[outcome] = {
                "edge": edge,
                "kelly_fraction": kelly_fraction,
                "stake_fraction": stake_fraction,
                "stake_amount": stake_amount
            }
    return suggestions

def poisson_prediction(home_team: str, away_team: str, stats_df: pd.DataFrame):
    """
    يتوقع احتمالات الفوز/التعادل/الخسارة باستخدام توزيع بواسون بناءً على متوسط الأهداف.
    """
    # حساب متوسطات الدوري
    avg_home_goals = stats_df['HomeGoals'].mean()
    avg_away_goals = stats_df['AwayGoals'].mean()

    # حساب قوة هجوم ودفاع كل فريق
    home_stats = stats_df[stats_df['HomeTeam'] == home_team]
    away_stats = stats_df[stats_df['AwayTeam'] == away_team]
    
    if home_stats.empty or away_stats.empty: return None

    home_attack_strength = home_stats['HomeGoals'].mean() / avg_home_goals
    home_defense_strength = home_stats['AwayGoals'].mean() / avg_away_goals
    away_attack_strength = away_stats['AwayGoals'].mean() / avg_away_goals
    away_defense_strength = away_stats['HomeGoals'].mean() / avg_home_goals

    # حساب الأهداف المتوقعة
    expected_home_goals = home_attack_strength * away_defense_strength * avg_home_goals
    expected_away_goals = away_attack_strength * home_defense_strength * avg_away_goals
    
    max_goals = 6
    home_goal_probs = [(math.exp(-expected_home_goals) * expected_home_goals**i) / math.factorial(i) for i in range(max_goals + 1)]
    away_goal_probs = [(math.exp(-expected_away_goals) * expected_away_goals**i) / math.factorial(i) for i in range(max_goals + 1)]
    
    home_win, draw, away_win = 0, 0, 0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = home_goal_probs[hg] * away_goal_probs[ag]
            if hg > ag: home_win += prob
            elif hg == ag: draw += prob
            else: away_win += prob
            
    total_prob = home_win + draw + away_win
    if total_prob == 0: return None
    return {"home": home_win/total_prob, "draw": draw/total_prob, "away": away_win/total_prob}

def calculate_form_probs(team_name: str, opponent_name: str, stats_df: pd.DataFrame, num_matches=6):
    """
    يحسب احتمالات الفوز بناءً على أداء الفريقين في آخر N مباريات (النقاط لكل مباراة).
    """
    team_matches = stats_df[(stats_df['HomeTeam'] == team_name) | (stats_df['AwayTeam'] == team_name)].tail(num_matches)
    opponent_matches = stats_df[(stats_df['HomeTeam'] == opponent_name) | (stats_df['AwayTeam'] == opponent_name)].tail(num_matches)

    if team_matches.empty or opponent_matches.empty: return None

    def get_ppg(df, team):
        points = 0
        for _, row in df.iterrows():
            if row['HomeTeam'] == team and row['Result'] == 'H': points += 3
            elif row['AwayTeam'] == team and row['Result'] == 'A': points += 3
            elif row['Result'] == 'D': points += 1
        return points / len(df) if not df.empty else 0

    team_ppg = get_ppg(team_matches, team_name)
    opponent_ppg = get_ppg(opponent_matches, opponent_name)
    
    total_ppg = team_ppg + opponent_ppg
    if total_ppg == 0: return {"home": 0.333, "draw": 0.333, "away": 0.333}

    home_prob = team_ppg / total_ppg
    away_prob = opponent_ppg / total_ppg
    draw_prob = 1 - abs(home_prob - away_prob)
    
    home_prob *= (1 - draw_prob)
    away_prob *= (1 - draw_prob)

    total_final = home_prob + draw_prob + away_prob
    if total_final == 0: return None
    return {"home": home_prob/total_final, "draw": draw_prob/total_final, "away": away_prob/total_final}

def calculate_xg_probs(home_team: str, away_team: str, stats_df: pd.DataFrame):
    """
    يحسب احتمالات الفوز باستخدام متوسط الأهداف المتوقعة (xG).
    """
    home_data = stats_df[stats_df['HomeTeam'] == home_team]
    away_data = stats_df[stats_df['AwayTeam'] == away_team]

    if home_data.empty or away_data.empty or 'Home_xG' not in stats_df.columns:
        return None

    # حساب متوسطات xG للدوري
    avg_home_xg = stats_df['Home_xG'].mean()
    avg_away_xg = stats_df['Away_xG'].mean()

    # قوة هجوم ودفاع كل فريق بناءً على xG
    home_attack_xg_strength = home_data['Home_xG'].mean() / avg_home_xg
    home_defense_xg_strength = home_data['Away_xG_for_Home'].mean() / avg_away_xg
    away_attack_xg_strength = away_data['Away_xG'].mean() / avg_away_xg
    away_defense_xg_strength = away_data['Home_xG_for_Away'].mean() / avg_home_xg

    # استخدام نموذج بواسون لكن بمعطيات xG
    expected_home_goals = home_attack_xg_strength * away_defense_xg_strength * avg_home_xg
    expected_away_goals = away_attack_xg_strength * home_defense_xg_strength * avg_away_xg
    
    max_goals = 6
    home_goal_probs = [(math.exp(-expected_home_goals) * expected_home_goals**i) / math.factorial(i) for i in range(max_goals + 1)]
    away_goal_probs = [(math.exp(-expected_away_goals) * expected_away_goals**i) / math.factorial(i) for i in range(max_goals + 1)]
    
    home_win, draw, away_win = 0, 0, 0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = home_goal_probs[hg] * away_goal_probs[ag]
            if hg > ag: home_win += prob
            elif hg == ag: draw += prob
            else: away_win += prob

    total_prob = home_win + draw + away_win
    if total_prob == 0: return None
    return {"home": home_win/total_prob, "draw": draw/total_prob, "away": away_win/total_prob}
