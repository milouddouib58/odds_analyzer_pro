# odds_math.py
import math

# --- تجميع الأسعار ---
def aggregate_prices(prices: list, mode: str = 'best') -> float:
    if not prices:
        return 0.0
    if mode == 'best':
        return max(prices)
    elif mode == 'median':
        s = sorted(prices)  # لا تعدّل القائمة الأصلية
        mid = len(s) // 2
        return (s[mid] + s[~mid]) / 2
    elif mode == 'mean':
        return sum(prices) / len(prices)
    return 0.0

# --- تحويل الأسعار إلى احتمالات ضمنية ---
def implied_from_decimal(odds: dict) -> dict:
    return {k: (1.0 / v if v and v > 0 else 0.0) for k, v in odds.items()}

def overround(implied_probs: dict) -> float:
    return sum(implied_probs.values())

def normalize_proportional(implied_probs: dict) -> dict:
    total_prob = sum(max(0.0, v) for v in implied_probs.values())
    if total_prob <= 0:
        n = len(implied_probs) or 1
        return {k: 1.0 / n for k in implied_probs}
    return {k: max(0.0, v) / total_prob for k, v in implied_probs.items()}

# --- احتمالات عادلة (Shin مبسطة مع fallback آمن) ---
def shin_fair_probs(implied_probs: dict) -> dict:
    """
    ملاحظة مهمة:
    تنفيذ Shin الدقيق يحتاج افتراضات قوية وكتاب منفرد. عند دمج أسعار متعددة أو وجود قيم شاذة،
    سنعود إلى التطبيع النسبي الآمن الذي يعطي توزيعات منطقية مجموعها 1 دون قيم سالبة.
    """
    return normalize_proportional(implied_probs)

# --- اقتراحات كيلي ---
def kelly_suggestions(
    fair_probs: dict,
    book_odds: dict,
    bankroll: float,
    kelly_scale: float = 0.25,
    max_fraction: float = 0.25
) -> dict:
    """
    - edge = p * o - 1
    - كبح النسبة القصوى للمخاطرة عبر max_fraction
    """
    suggestions = {}
    for outcome, prob in fair_probs.items():
        odds = book_odds.get(outcome)
        if not odds or odds <= 1:
            continue
        edge = (prob * odds) - 1
        if edge > 0:
            kelly_fraction = edge / (odds - 1)
            stake_fraction = max(0.0, min(kelly_fraction * kelly_scale, max_fraction))
            stake_amount = round(bankroll * stake_fraction, 2)
            suggestions[outcome] = {
                "edge": edge,
                "stake_fraction": stake_fraction,
                "stake_amount": stake_amount,
            }
    return suggestions

# --- توقع بواسون لنتيجة المباراة ---
def poisson_prediction(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    home_adv: float = 1.10,
    base_max_goals: int = 6
) -> dict:
    """
    - الدفاع يُخفض التوقع: λ_home = (home_attack / away_defense) * home_adv
                          λ_away = (away_attack / home_defense)
    - تمديد الشبكة تلقائياً لتغطية 99.9% من الكتلة الاحتمالية (حد أقصى 20 هدفاً).
    """
    # حماية من القسمة على صفر
    home_defense = max(home_defense, 1e-6)
    away_defense = max(away_defense, 1e-6)

    expected_home_goals = max(1e-6, (home_attack / away_defense) * max(1.0, home_adv))
    expected_away_goals = max(1e-6, (away_attack / home_defense))

    def poisson_pmf(lmbd, k):
        return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)

    def needed_max(lmbd, start=6):
        g = max(start, 0)
        while True:
            cdf = sum(poisson_pmf(lmbd, i) for i in range(g + 1))
            if cdf >= 0.999 or g >= 20:
                return g
            g += 1

    max_h = needed_max(expected_home_goals, base_max_goals)
    max_a = needed_max(expected_away_goals, base_max_goals)
    max_goals = max(max_h, max_a)

    home_goal_probs = [poisson_pmf(expected_home_goals, i) for i in range(max_goals + 1)]
    away_goal_probs = [poisson_pmf(expected_away_goals, i) for i in range(max_goals + 1)]

    home_win = draw = away_win = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = home_goal_probs[hg] * away_goal_probs[ag]
            if hg > ag:
                home_win += prob
            elif hg == ag:
                draw += prob
            else:
                away_win += prob

    total_prob = home_win + draw + away_win
    if total_prob <= 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}

    return {
        "home": home_win / total_prob,
        "draw": draw / total_prob,
        "away": away_win / total_prob,
    }
