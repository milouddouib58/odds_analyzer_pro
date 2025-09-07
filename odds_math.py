# odds_math.py
import math

def aggregate_prices(prices: list, mode: str = 'median') -> float:
    if not prices: return 0.0
    if mode == 'median':
        prices.sort()
        mid = len(prices) // 2
        return (prices[mid] + prices[~mid]) / 2
    elif mode == 'best':
        return max(prices)
    elif mode == 'mean':
        return sum(prices) / len(prices)
    return 0.0

def implied_from_decimal(odds: dict) -> dict:
    return {k: 1/v if v > 0 else 0 for k, v in odds.items()}

def overround(implied_probs: dict) -> float:
    return sum(implied_probs.values())

def normalize_proportional(implied_probs: dict) -> dict:
    total_prob = overround(implied_probs)
    if total_prob == 0: return implied_probs
    return {k: v / total_prob for k, v in implied_probs.items()}

def shin_fair_probs(implied_probs: dict) -> dict:
    # طريقة Shin لإزالة الهامش
    if not implied_probs: return {}
    n = len(implied_probs)
    if n == 0: return {}
    
    def sum_diff_sq(z, probs):
        return sum(((p - z) / (1 - z if p < z else p))**2 for p in probs) - 1

    implied_values = list(implied_probs.values())
    
    # البحث عن قيمة z
    low = 0.0
    high = min(implied_values)
    z = 0.0
    
    for _ in range(100): # 100 تكرار يكفي للدقة
        z = (low + high) / 2
        if sum_diff_sq(z, implied_values) > 0:
            low = z
        else:
            high = z
            
    total_prob = sum(implied_values)
    if total_prob < 1.0: # إذا كان السوق تحت 100%، لا يوجد هامش
        return normalize_proportional(implied_probs)

    fair_probs_values = [(p - z) / (1 - z) for p in implied_values]
    return dict(zip(implied_probs.keys(), fair_probs_values))

def kelly_suggestions(fair_probs: dict, book_odds: dict, bankroll: float, kelly_scale: float = 0.25) -> dict:
    suggestions = {}
    for outcome, prob in fair_probs.items():
        odds = book_odds.get(outcome)
        if not odds or odds <= 1:
            continue
        
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
