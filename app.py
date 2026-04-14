# -*- coding: utf-8 -*-
import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

st.set_page_config(
    page_title="🧠 Mastermind PRO",
    page_icon="🕵️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;900&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }
body { background: #0a0a0f; }
.main { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%); }

.match-card {
    background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(20,20,35,0.98));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,215,0,0.15);
    box-shadow: 0 20px 60px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,215,0,0.1);
    border-radius: 24px;
    padding: 40px;
    margin-bottom: 30px;
    direction: rtl;
    color: white;
    position: relative;
    overflow: hidden;
}
.match-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #FFD700, #FFA500, #FFD700, transparent);
}
.match-title {
    font-size: 28px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.match-subtitle {
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 13px;
    margin-bottom: 0;
    letter-spacing: 1px;
}
.rec-golden {
    background: linear-gradient(135deg, #1a1200, #2a1f00);
    border: 2px solid #FFD700;
    box-shadow: 0 0 30px rgba(255,215,0,0.35);
    padding: 22px;
    border-radius: 16px;
    margin-top: 10px;
    color: white;
    position: relative;
    overflow: hidden;
    animation: goldenPulse 2.5s ease-in-out infinite;
}
.rec-golden::after {
    content: '⭐ VALUE BET';
    position: absolute;
    top: 10px; left: 10px;
    background: #FFD700;
    color: #000;
    font-size: 9px;
    font-weight: 900;
    padding: 2px 7px;
    border-radius: 5px;
    letter-spacing: 1px;
}
@keyframes goldenPulse {
    0%,100% { box-shadow: 0 0 25px rgba(255,215,0,0.3); }
    50%      { box-shadow: 0 0 50px rgba(255,215,0,0.6); }
}
.rec-strong {
    background: linear-gradient(135deg, #001a0d, #002a15);
    border: 2px solid #00ff88;
    box-shadow: 0 0 20px rgba(0,255,136,0.2);
    padding: 22px; border-radius: 16px; margin-top: 10px; color: white;
}
.rec-value {
    background: linear-gradient(135deg, #0d0d1a, #12123a);
    border: 2px solid #4d79ff;
    box-shadow: 0 0 20px rgba(77,121,255,0.2);
    padding: 22px; border-radius: 16px; margin-top: 10px; color: white;
}
.rec-extra {
    background: linear-gradient(135deg, #0d1a1a, #0d2a2a);
    border: 2px solid #00bcd4;
    box-shadow: 0 0 15px rgba(0,188,212,0.15);
    padding: 22px; border-radius: 16px; margin-top: 10px; color: white;
}
.rec-label {
    font-size: 12px; font-weight: 700;
    letter-spacing: 1px; margin-bottom: 8px; opacity: 0.85;
}
.rec-bet-name {
    font-size: 20px; font-weight: 900; margin: 10px 0; line-height: 1.3;
}
.rec-odds-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 12px; border-radius: 20px;
    font-size: 18px; font-weight: 900; margin: 4px 0;
}
.rec-stake-highlight { font-size: 26px; font-weight: 900; color: #FFD700; }
.ev-badge-pos {
    display: inline-block;
    background: linear-gradient(90deg, #00c851, #007e33);
    color: white; font-size: 12px; font-weight: 700;
    padding: 3px 10px; border-radius: 10px; margin-top: 6px;
}
.ev-badge-neg {
    display: inline-block;
    background: rgba(255,50,50,0.2);
    color: rgba(255,150,150,0.8);
    font-size: 12px; font-weight: 700;
    padding: 3px 10px; border-radius: 10px; margin-top: 6px;
}
.narrative-box {
    background: linear-gradient(135deg, rgba(255,215,0,0.05), rgba(255,140,0,0.03));
    border: 1px solid rgba(255,215,0,0.3);
    border-right: 5px solid #FFD700;
    padding: 22px 28px; border-radius: 14px; margin: 20px 0;
    font-size: 16px; line-height: 1.9;
    color: rgba(255,255,255,0.9); direction: rtl; position: relative;
}
.narrative-box::before {
    content: '🧠'; position: absolute; top: -14px; right: 18px;
    font-size: 26px; background: #0d1117; padding: 0 4px;
}
.market-insight {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 12px 18px; margin: 6px 0;
    display: flex; justify-content: space-between;
    align-items: center; direction: rtl;
}
.insight-label { color: rgba(255,255,255,0.5); font-size: 13px; }
.insight-value { color: white; font-weight: 700; font-size: 15px; }
.sidebar-header {
    background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,140,0,0.05));
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 12px; padding: 14px; margin-bottom: 14px;
    text-align: center; color: #FFD700; font-weight: 700; font-size: 15px;
}
.no-recs {
    background: rgba(255,50,50,0.06);
    border: 1px solid rgba(255,50,50,0.18);
    border-radius: 10px; padding: 14px 18px;
    color: rgba(255,150,150,0.8); font-size: 14px;
    text-align: center; direction: rtl; margin-top: 10px;
}
.golden-opp-card {
    background: linear-gradient(135deg, rgba(255,215,0,0.07), rgba(255,140,0,0.03));
    border: 1px solid rgba(255,215,0,0.28);
    border-radius: 12px; padding: 14px 18px; margin: 7px 0; direction: rtl;
}
.warning-box {
    background: rgba(255,100,0,0.08);
    border: 1px solid rgba(255,100,0,0.25);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    color: rgba(255,180,100,0.9); font-size: 13px; direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# الثوابت
# ══════════════════════════════════════════════════════
API_BASE_URL  = "https://api.the-odds-api.com/v4"
MIN_STAKE     = 22.0
MAX_STAKE_PCT = 0.15
MIN_STAKE_PCT = 0.01
SHARP_BOOKS   = ["Pinnacle", "Betfair", "Matchbook", "SBOBet"]

# ══════════════════════════════════════════════════════
# جلب البيانات
# ✅ إصلاح: فصل الأخطاء عن الـ cache — الدالة تُعيد
#    tuple(data, error_msg) بدلاً من استدعاء st.error مباشرة
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_cached(api_key: str, sport_key: str) -> Tuple[List[dict], str, str]:
    """
    يُعيد: (data, error_message, api_remaining)
    data = [] عند الخطأ
    error_message = "" عند النجاح
    """
    url = f"{API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey":       api_key,
        "regions":      "eu,uk,us,au",
        "markets":      "h2h,totals",
        "oddsFormat":   "decimal",
        "includeLinks": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        remaining = resp.headers.get("x-requests-remaining", "?")
        if resp.status_code == 200:
            return resp.json(), "", remaining
        elif resp.status_code == 401:
            return [], "❌ API Key غير صالح أو منتهي الصلاحية", "?"
        elif resp.status_code == 422:
            return [], "⚠️ البطولة المختارة غير متاحة حالياً", "?"
        elif resp.status_code == 429:
            return [], "⚠️ تجاوزت حد الطلبات المسموح به", "?"
        else:
            return [], f"خطأ في الاتصال: {resp.status_code}", "?"
    except requests.Timeout:
        return [], "⏱️ انتهت مهلة الاتصال — حاول مرة أخرى", "?"
    except Exception as exc:
        return [], f"خطأ غير متوقع: {exc}", "?"


def fetch_odds(api_key: str, sport_key: str) -> List[dict]:
    """
    Wrapper يعرض الأخطاء في واجهة Streamlit
    ويحفظ api_remaining في session_state بأمان
    """
    data, error_msg, remaining = fetch_odds_cached(api_key, sport_key)
    # ✅ session_state خارج الـ cached function
    st.session_state["api_remaining"] = remaining
    if error_msg:
        st.error(error_msg)
    return data

# ══════════════════════════════════════════════════════
# دوال الرياضيات
# ══════════════════════════════════════════════════════
def remove_margin(odds_list: List[float]) -> Optional[List[float]]:
    """إزالة هامش البوكماكر — يُعيد الاحتمالات الحقيقية"""
    if not odds_list or any(o <= 1.0 for o in odds_list):
        return None
    probs = [1.0 / o for o in odds_list]
    total = sum(probs)
    if total <= 0:
        return None
    return [p / total for p in probs]


def calculate_vig(odds_list: List[float]) -> float:
    """هامش البوكماكر كنسبة مئوية"""
    if not odds_list:
        return 0.0
    probs = [1.0 / o for o in odds_list if o > 1.0]
    return (sum(probs) - 1.0) * 100


def calculate_ev(true_prob: float, odds: float) -> float:
    """
    القيمة المتوقعة
    양수(موجب) = ربح على المدى البعيد  ✅ إصلاح: أُزيلت الكلمة الكورية
    """
    return (true_prob * (odds - 1.0)) - (1.0 - true_prob)


def kelly_criterion(true_prob: float, odds: float, fraction: float) -> float:
    """كيلي الكسري لتحديد حجم الرهان الأمثل"""
    b = odds - 1.0
    q = 1.0 - true_prob
    if b <= 0 or true_prob <= 0:
        return 0.0
    kelly = (b * true_prob - q) / b
    return max(0.0, kelly * fraction)


def calculate_stake(
    true_prob: float, odds: float, bankroll: float, fraction: float
) -> float:
    """المبلغ المقترح مع حدود الأمان"""
    kf = kelly_criterion(true_prob, odds, fraction)
    if kf <= 0:
        return 0.0
    raw    = bankroll * kf
    capped = min(raw, bankroll * MAX_STAKE_PCT)
    return max(capped, bankroll * MIN_STAKE_PCT, MIN_STAKE)


def detect_line_value(
    true_prob: float, book_odds: float, threshold: float = 0.03
) -> Dict[str, Any]:
    """كشف خطأ التسعير"""
    if book_odds <= 1.0 or true_prob <= 0:
        return {
            "has_value": False, "ev": -1.0,
            "edge_pct": 0.0, "confidence": 0, "implied_prob": 1.0
        }
    ev           = calculate_ev(true_prob, book_odds)
    implied_prob = 1.0 / book_odds
    edge         = true_prob - implied_prob
    has_value    = (ev > threshold) and (edge > 0)
    confidence   = min(100, max(0, int(abs(ev) * 200 + edge * 150)))
    return {
        "has_value":    has_value,
        "ev":           ev,
        "edge_pct":     edge * 100,
        "confidence":   confidence,
        "implied_prob": implied_prob,
    }

# ══════════════════════════════════════════════════════
# دوال السوق
# ══════════════════════════════════════════════════════
def get_sharp_consensus(
    bookmakers: List[dict], market_key: str, point: Optional[float] = None
) -> Dict[str, Any]:
    """
    إجماع السوق الحاد:
    1) يبحث عن Sharp Books بالترتيب
    2) ثم متوسط مرجّح لكل الكتب
    """
    def filter_outcomes(outcomes: list) -> list:
        if market_key == "totals" and point is not None:
            return [o for o in outcomes if abs(o.get("point", 0) - point) < 0.01]
        return outcomes

    # ── مرحلة 1: Sharp Books ──
    for book_name in SHARP_BOOKS:
        for b in bookmakers:
            if b.get("title", "").lower() != book_name.lower():
                continue
            for m in b.get("markets", []):
                if m["key"] != market_key:
                    continue
                outcomes = filter_outcomes(m.get("outcomes", []))
                if len(outcomes) < 2:
                    continue
                odds_list  = [o["price"] for o in outcomes]
                true_probs = remove_margin(odds_list)
                if not true_probs:
                    continue
                return {
                    "outcomes":    [o["name"] for o in outcomes],
                    "true_probs":  true_probs,
                    "raw_odds":    odds_list,
                    "vig":         calculate_vig(odds_list),
                    "book_used":   book_name,
                    "n_books":     1,
                    "is_sharp":    True,   # ✅ علامة لتمييز Sharp Books
                    "odds_spread": max(odds_list) - min(odds_list)
                                   if len(odds_list) >= 2 else 0.0,
                }

    # ── مرحلة 2: متوسط السوق ──
    all_probs:    Dict[str, List[float]] = {}
    all_raw_odds: Dict[str, List[float]] = {}
    count_books = 0

    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] != market_key:
                continue
            outcomes = filter_outcomes(m.get("outcomes", []))
            if len(outcomes) < 2:
                continue
            odds_list  = [o["price"] for o in outcomes]
            true_probs = remove_margin(odds_list)
            if not true_probs:
                continue
            count_books += 1
            for i, o in enumerate(outcomes):
                key = o["name"]
                all_probs.setdefault(key, []).append(true_probs[i])
                all_raw_odds.setdefault(key, []).append(odds_list[i])

    if all_probs and count_books >= 2:
        names     = list(all_probs.keys())
        avg_probs = [float(np.mean(all_probs[k])) for k in names]
        total     = sum(avg_probs)
        norm      = [p / total for p in avg_probs] if total > 0 else avg_probs
        avg_odds  = [
            float(np.mean(all_raw_odds[k])) if all_raw_odds.get(k) else 2.0
            for k in names
        ]
        return {
            "outcomes":    names,
            "true_probs":  norm,
            "raw_odds":    avg_odds,
            "vig":         calculate_vig(avg_odds),
            "book_used":   f"إجماع {count_books} كتاب",
            "n_books":     count_books,
            "is_sharp":    False,
            "odds_spread": max(avg_odds) - min(avg_odds)
                           if len(avg_odds) >= 2 else 0.0,
        }
    return {}


def get_target_book_odds(
    bookmakers: List[dict], market_key: str,
    target_book: str, point: Optional[float] = None
) -> Dict[str, float]:
    """أودز الشركة المستهدفة فقط"""
    for b in bookmakers:
        if b.get("title", "").lower() != target_book.lower():
            continue
        for m in b.get("markets", []):
            if m["key"] != market_key:
                continue
            outcomes = m.get("outcomes", [])
            if market_key == "totals" and point is not None:
                outcomes = [
                    o for o in outcomes
                    if abs(o.get("point", 0) - point) < 0.01
                ]
            if outcomes:
                return {o["name"]: o["price"] for o in outcomes}
    return {}


def get_available_totals_points(bookmakers: List[dict]) -> List[float]:
    points: set = set()
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] == "totals":
                for o in m.get("outcomes", []):
                    if "point" in o:
                        points.add(float(o["point"]))
    return sorted(list(points))


def market_consensus_label(n_books: int, vig: float, is_sharp: bool) -> str:
    """
    ✅ إصلاح: Pinnacle وحده (is_sharp=True) = "مرجع حاد ✅"
    وليس "ضعيف" بسبب n_books=1
    """
    if is_sharp:
        return "مرجع حاد (Sharp) ✅"
    if n_books >= 8 and vig < 5:
        return "إجماع قوي جداً ✅"
    elif n_books >= 5 and vig < 8:
        return "إجماع جيد 🟡"
    elif n_books >= 3:
        return "إجماع مقبول ⚠️"
    return "إجماع ضعيف ❌"

# ══════════════════════════════════════════════════════
# بناء التوصيات — المنطق المُصحح بالكامل
# ══════════════════════════════════════════════════════
def _classify_h2h_candidate(
    team_name: str,
    true_prob: float,
    book_odds: float,
    ev: float,
    edge: float,
    vi: Dict[str, Any],
    stake: int,
    min_ev_threshold: float,
    user_bankroll: float,
) -> Optional[Dict[str, Any]]:
    """
    تصنيف مرشح H2H
    ✅ إصلاح: إزالة الـ else الذي كان يُسقط حالات EV موجب
    """
    if ev >= min_ev_threshold and edge >= 0.03:
        label = "💎 خطأ تسعير مكتشف!"
        tier  = "golden"
        stake = max(
            round(min(stake * 1.25, user_bankroll * MAX_STAKE_PCT)),
            int(MIN_STAKE),
        )
    elif ev > 0 and true_prob >= 0.58 and book_odds < 1.75:
        label = "🟢 مفضل بقيمة موثوقة"
        tier  = "strong"
    elif ev > 0:
        # ✅ إصلاح: أي EV موجب لا يقع في الفئتين أعلاه → "قيمة جيدة"
        #    بدلاً من الوقوع في else والتجاهل
        label = "🔵 قيمة جيدة"
        tier  = "value"
    else:
        return None  # EV سالب أو صفر — تجاهل

    return {
        "bet":          f"فوز {team_name}",
        "team":         team_name,
        "odds":         book_odds,
        "true_prob":    true_prob,
        "implied_prob": vi["implied_prob"],
        "stake":        stake,
        "ev":           ev,
        "edge_pct":     edge * 100,
        "confidence":   vi["confidence"],
        "market":       "h2h",
        "label":        label,
        "tier":         tier,
    }


def _classify_ou_candidate(
    ou_label_ar: str,
    ou_true_p: float,
    book_ou_odds: float,
    ev_ou: float,
    edge_ou: float,
    vi_ou: Dict[str, Any],
    stake_ou: int,
    min_ev_threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    تصنيف مرشح O/U
    ✅ إصلاح: نفس منطق H2H — أي EV موجب يُقبل
    """
    if ev_ou >= min_ev_threshold and edge_ou >= 0.03:
        label_ou = "💎 قيمة في الأهداف!"
        tier_ou  = "golden"
        stake_ou = max(round(stake_ou * 1.2), int(MIN_STAKE))
    elif ev_ou > 0:
        # ✅ إصلاح: EV موجب لكن أقل من الحد → "مثير للاهتمام" بدلاً من التجاهل
        label_ou = "🟡 أهداف مثيرة للاهتمام"
        tier_ou  = "extra"
    else:
        return None  # EV سالب — تجاهل

    return {
        "bet":          ou_label_ar,
        "odds":         book_ou_odds,
        "true_prob":    ou_true_p,
        "implied_prob": vi_ou["implied_prob"],
        "stake":        stake_ou,
        "ev":           ev_ou,
        "edge_pct":     edge_ou * 100,
        "confidence":   vi_ou["confidence"],
        "market":       "totals",
        "label":        label_ou,
        "tier":         tier_ou,
    }


def build_recommendations(
    home_team: str,
    away_team: str,
    h_true: float,
    a_true: float,
    h_odds_raw: float,
    a_odds_raw: float,
    ou_over_true: float,
    ou_under_true: float,
    target_ou: Dict[str, float],
    best_point: float,
    user_bankroll: float,
    active_fraction: float,
    min_ev_threshold: float,
    min_odds_filter: float,
    max_odds_filter: float,
) -> List[Dict[str, Any]]:
    """
    قواعد صارمة:
    ✅ EV إيجابي حقيقي لكل توصية
    ✅ رهان واحد فقط من H2H (الأفضل EV)
    ✅ رهان واحد فقط من O/U  (الأفضل EV)
    ✅ لا توصية بفريقين متعاكسين في نفس المباراة
    ✅ الحد الأقصى: توصيتان لكل مباراة
    """

    # ════════════════════════════════
    # A. تحليل H2H
    # ════════════════════════════════
    h2h_candidates: List[Dict[str, Any]] = []

    for team_name, true_prob, book_odds in [
        (home_team, h_true, h_odds_raw),
        (away_team, a_true, a_odds_raw),
    ]:
        if book_odds <= 1.0 or true_prob <= 0.05 or true_prob >= 0.98:
            continue
        if not (min_odds_filter <= book_odds <= max_odds_filter):
            continue

        ev           = calculate_ev(true_prob, book_odds)
        implied_prob = 1.0 / book_odds
        edge         = true_prob - implied_prob

        if ev <= 0:
            continue

        vi    = detect_line_value(true_prob, book_odds, min_ev_threshold)
        stake = max(
            round(calculate_stake(true_prob, book_odds, user_bankroll, active_fraction)),
            int(MIN_STAKE),
        )

        candidate = _classify_h2h_candidate(
            team_name, true_prob, book_odds,
            ev, edge, vi, stake,
            min_ev_threshold, user_bankroll,
        )
        if candidate:
            h2h_candidates.append(candidate)

    # ✅ رهان واحد فقط من H2H
    best_h2h: List[Dict[str, Any]] = []
    if h2h_candidates:
        best_h2h = [max(h2h_candidates, key=lambda x: x["ev"])]

    # ════════════════════════════════
    # B. تحليل O/U
    # ════════════════════════════════
    ou_candidates: List[Dict[str, Any]] = []

    over_key  = next((k for k in target_ou if "over"  in k.lower()), None)
    under_key = next((k for k in target_ou if "under" in k.lower()), None)

    for ou_key, ou_true_p, ou_label_ar in [
        (over_key,  ou_over_true,  f"أكثر من {best_point} أهداف"),
        (under_key, ou_under_true, f"أقل من {best_point} أهداف"),
    ]:
        if not ou_key:
            continue
        book_ou_odds = target_ou.get(ou_key, 0.0)
        if book_ou_odds <= 1.0 or ou_true_p <= 0.05:
            continue
        if not (min_odds_filter <= book_ou_odds <= max_odds_filter):
            continue

        ev_ou   = calculate_ev(ou_true_p, book_ou_odds)
        impl_ou = 1.0 / book_ou_odds
        edge_ou = ou_true_p - impl_ou

        if ev_ou <= 0:
            continue

        vi_ou    = detect_line_value(ou_true_p, book_ou_odds, min_ev_threshold)
        stake_ou = max(
            round(calculate_stake(
                ou_true_p, book_ou_odds, user_bankroll, active_fraction
            )),
            int(MIN_STAKE),
        )

        candidate_ou = _classify_ou_candidate(
            ou_label_ar, ou_true_p, book_ou_odds,
            ev_ou, edge_ou, vi_ou, stake_ou,
            min_ev_threshold,
        )
        if candidate_ou:
            ou_candidates.append(candidate_ou)

    # ✅ رهان واحد فقط من O/U
    best_ou: List[Dict[str, Any]] = []
    if ou_candidates:
        best_ou = [max(ou_candidates, key=lambda x: x["ev"])]

    # ════════════════════════════════
    # C. دمج وترتيب (أقصى توصيتان)
    # ════════════════════════════════
    tier_order = {"golden": 0, "strong": 1, "value": 2, "extra": 3}
    final = sorted(
        best_h2h + best_ou,
        key=lambda x: (-x["ev"], tier_order.get(x.get("tier", "extra"), 4)),
    )
    return final[:2]

# ══════════════════════════════════════════════════════
# السرد الذكي
# ✅ إصلاح: إزالة d_prob من المعاملات (كان مُمرَّراً لكن غير مستخدم)
# ✅ إصلاح: fav_home لا تُعيَّن None — تُعالج بشكل منفصل
# ══════════════════════════════════════════════════════
def generate_narrative(
    home_team: str,
    away_team: str,
    h_prob: float,
    a_prob: float,
    ou_prob_over: float,
    h2h_consensus: Dict[str, Any],
    target_book: str,
    recs: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    # ✅ إصلاح: تحديد المفضل بشكل واضح مع معالجة صحيحة لـ fav_home
    fav: Optional[str]      = None
    underdog: Optional[str] = None
    fav_prob  = 0.0
    dog_prob  = 0.0
    fav_home_str = ""  # ✅ نص جاهز بدلاً من bool قد يكون None

    if h_prob > a_prob + 0.15:
        fav, underdog = home_team, away_team
        fav_prob, dog_prob = h_prob, a_prob
        fav_home_str = "أمام جمهوره"
    elif a_prob > h_prob + 0.15:
        fav, underdog = away_team, home_team
        fav_prob, dog_prob = a_prob, h_prob
        fav_home_str = "خارج أرضه"

    is_balanced  = abs(h_prob - a_prob) < 0.12
    is_one_sided = abs(h_prob - a_prob) > 0.30
    high_scoring = ou_prob_over > 0.56
    low_scoring  = ou_prob_over < 0.44
    n_books      = h2h_consensus.get("n_books", 0)
    vig          = h2h_consensus.get("vig", 0.0)
    is_sharp     = h2h_consensus.get("is_sharp", False)

    if is_sharp:
        lines.append(
            f"🎯 **مرجع حاد (Sharp Book):** البيانات مستخرجة مباشرة من "
            f"Pinnacle — أدق مصدر للاحتمالات الحقيقية."
        )
    elif n_books >= 7:
        lines.append(
            f"📊 **إجماع سوقي قوي:** {n_books} شركة تُسعّر المباراة "
            f"بهامش {vig:.1f}٪ — الأرقام موثوقة."
        )

    if fav and is_one_sided:
        lines.append(
            f"⚡ **هيمنة واضحة:** {fav} يتقدم بـ {fav_prob*100:.0f}٪ احتمالاً "
            f"{fav_home_str}. {underdog} يصعب عليه المنافسة."
        )
        if high_scoring:
            lines.append("🔥 **هجوم نشط:** مؤشرات السوق تدعم مباراة بأهداف كثيرة.")
        elif low_scoring:
            lines.append("🔒 **دفاع متين:** نتيجة ضيقة هي السيناريو الأرجح.")
    elif fav:
        lines.append(
            f"📌 **أفضلية معتدلة:** {fav} ({fav_prob*100:.0f}٪) أمام "
            f"{underdog} ({dog_prob*100:.0f}٪) — ليست حاسمة."
        )
    elif is_balanced:
        lines.append(
            "⚖️ **توازن تام:** السوق عاجز عن الترجيح — "
            "تجنب الرهان على النتيجة."
        )

    if ou_prob_over > 0.60:
        lines.append(
            f"📈 **ضغط هجومي:** {ou_prob_over*100:.0f}٪ لتخطي 2.5 أهداف."
        )
    elif ou_prob_over < 0.40:
        lines.append(
            f"📉 **معركة دفاعية:** "
            f"{(1 - ou_prob_over)*100:.0f}٪ للبقاء تحت 2.5 أهداف."
        )

    golden = [r for r in recs if r.get("tier") == "golden"]
    if golden:
        for gr in golden:
            fair_odds = (
                f"{1 / gr['true_prob']:.2f}" if gr["true_prob"] > 0 else "—"
            )
            lines.append(
                f"💎 **خطأ تسعير!** {target_book} تعرض '{gr['bet']}' "
                f"بأودز {gr['odds']:.2f} والأودز العادل {fair_odds}. "
                f"ميزة: **+{gr['ev']*100:.1f}٪**"
            )

    if is_balanced and not golden:
        lines.append(
            "🚫 **تحذير:** مباراة متكافئة بلا قيمة واضحة — "
            "يُنصح بالتجاهل."
        )
    elif fav and is_one_sided and golden:
        lines.append(
            f"🎯 **الخلاصة:** هيمنة {fav} + خطأ تسعير "
            f"{target_book} = فرصة نادرة."
        )
    elif fav and not golden:
        lines.append(
            f"💡 **نصيحة:** {fav} في أكومولاتور أفضل من الرهان المنفرد."
        )

    return (
        "\n\n".join(lines)
        if lines
        else "🔍 البيانات غير كافية لتوليد قراءة تحليلية موثوقة."
    )

# ══════════════════════════════════════════════════════
# دوال الرسم البياني
# ══════════════════════════════════════════════════════
def render_prob_chart(
    home_team: str, away_team: str,
    h_true: float, a_true: float, d_true: float,
    h_implied: float, a_implied: float, d_implied: float,
    target_book: str, chart_key: str,
) -> None:
    cats      = [home_team[-12:], "تعادل", away_team[-12:]]
    true_vals = [round(h_true * 100, 1), round(d_true * 100, 1), round(a_true * 100, 1)]
    impl_vals = [
        round(h_implied * 100, 1),
        round(d_implied * 100, 1),
        round(a_implied * 100, 1),
    ]
    impl_vals = [max(0.0, v) for v in impl_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="الاحتمال الحقيقي",
        x=cats, y=true_vals,
        marker_color=[
            "rgba(255,215,0,0.85)",
            "rgba(160,160,160,0.6)",
            "rgba(77,121,255,0.85)",
        ],
        marker_line_color=["#FFD700", "#aaa", "#4d79ff"],
        marker_line_width=2,
        text=[f"{v}%" for v in true_vals],
        textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig.add_trace(go.Bar(
        name=f"ضمني {target_book}",
        x=cats, y=impl_vals,
        marker_color=[
            "rgba(255,100,100,0.4)",
            "rgba(120,120,120,0.3)",
            "rgba(100,220,100,0.4)",
        ],
        marker_line_color=["#ff6464", "#888", "#64dd64"],
        marker_line_width=1,
        text=[f"{v}%" for v in impl_vals],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.55)", size=10),
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=290,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.15)", borderwidth=1,
        ),
        margin=dict(l=8, r=8, t=38, b=8),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="white")),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            ticksuffix="%",
            tickfont=dict(color="rgba(255,255,255,0.4)"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_ev_gauge(ev: float, confidence: int, gauge_key: str) -> None:
    color  = "#00ff88" if ev > 0 else "#ff4444"
    ev_pct = ev * 100
    fig    = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ev_pct,
        delta={"reference": 0, "valueformat": ".1f", "suffix": "%"},
        title={
            "text": (
                f"القيمة المتوقعة (EV)<br>"
                f"<span style='font-size:11px;color:rgba(255,255,255,0.45)'>"
                f"Confidence: {confidence}%</span>"
            ),
            "font": {"size": 14, "color": "white", "family": "Tajawal"},
        },
        number={
            "suffix": "%",
            "font": {"size": 26, "color": color},
            "valueformat": ".1f",
        },
        gauge={
            "axis": {
                "range": [-30, 30],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.25)",
                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 9},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0.3)",
            "borderwidth": 0,
            "steps": [
                {"range": [-30, -10], "color": "rgba(255,50,50,0.12)"},
                {"range": [-10,   0], "color": "rgba(255,165,0,0.08)"},
                {"range": [  0,  10], "color": "rgba(0,255,100,0.08)"},
                {"range": [ 10,  30], "color": "rgba(0,255,100,0.22)"},
            ],
            "threshold": {
                "line": {"color": "#FFD700", "width": 3},
                "thickness": 0.75,
                "value": 5,
            },
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Tajawal"),
    )
    st.plotly_chart(fig, use_container_width=True, key=gauge_key)


def render_vig_chart(bookmakers: List[dict], chart_key: str) -> None:
    vigs: Dict[str, float] = {}
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] == "h2h":
                ods = [
                    o["price"] for o in m.get("outcomes", []) if o["price"] > 1.0
                ]
                if ods:
                    vigs[b["title"]] = calculate_vig(ods)

    if len(vigs) < 3:
        st.caption("⚠️ بيانات غير كافية لعرض رادار الهوامش (أقل من 3 شركات)")
        return

    sorted_v = sorted(vigs.items(), key=lambda x: x[1])[:12]
    names    = [v[0] for v in sorted_v]
    values   = [v[1] for v in sorted_v]

    # ✅ إصلاح: التحقق من values قبل max()
    if not values:
        return

    colors = [
        "#00ff88" if v < 4 else "#FFD700" if v < 7 else "#ff6464"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.08)",
        marker_line_width=1,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(
        title=dict(
            text="🔬 هوامش شركات المراهنة (الأقل أفضل للاعب)",
            font=dict(size=12, color="rgba(255,255,255,0.65)", family="Tajawal"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=max(240, len(names) * 26),
        margin=dict(l=8, r=45, t=36, b=8),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            ticksuffix="%",
            # ✅ إصلاح: max(values) آمن بعد التحقق أعلاه
            range=[0, max(values) * 1.3],
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# ══════════════════════════════════════════════════════
# الشريط الجانبي
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-header">🧠 MASTERMIND PRO<br>'
        '<small style="font-weight:300;opacity:0.65">'
        'نظام تحليل الأودز المتقدم</small></div>',
        unsafe_allow_html=True,
    )

    api_key_input = st.text_input(
        "🔑 Odds API Key",
        type="password",
        placeholder="أدخل مفتاح API الخاص بك",
    )
    if "api_remaining" in st.session_state:
        remaining_val = st.session_state["api_remaining"]
        st.caption(f"📡 طلبات متبقية: **{remaining_val}**")

    st.markdown("---")
    target_bookmaker = st.radio(
        "🎯 منصة الرهان المستهدفة",
        ["1xBet", "Melbet", "Bet365", "Unibet", "William Hill"],
        help="الشركة التي ستراهن عليها فعلياً",
    )

    st.markdown("---")
    SPORTS_MAP: Dict[str, str] = {
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 الدوري الإنجليزي":  "soccer_epl",
        "🇪🇸 الدوري الإسباني":          "soccer_spain_la_liga",
        "🇩🇪 الدوري الألماني":          "soccer_germany_bundesliga",
        "🇮🇹 الدوري الإيطالي":          "soccer_italy_serie_a",
        "🇫🇷 الدوري الفرنسي":          "soccer_france_ligue_one",
        "🏆 دوري أبطال أوروبا":        "soccer_uefa_champs_league",
        "🌍 الدوري الأوروبي":          "soccer_uefa_europa_league",
        "🏀 NBA كرة السلة":            "basketball_nba",
        "🎾 تنس ATP":                   "tennis_atp",
    }
    selected_sport_label = st.selectbox(
        "🏆 اختر البطولة", list(SPORTS_MAP.keys())
    )
    sport_key = SPORTS_MAP[selected_sport_label]

    st.markdown("---")
    st.subheader("💰 إدارة الرأسمال")
    user_bankroll = st.number_input(
        "💵 الرصيد الحالي (وحدة)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=1_000.0,
        step=100.0,
    )
    risk_profile = st.select_slider(
        "📊 مستوى المخاطرة",
        options=["محافظ جداً", "محافظ", "متوازن", "جريء", "مغامر"],
        value="متوازن",
    )
    kelly_map: Dict[str, float] = {
        "محافظ جداً": 0.10,
        "محافظ":      0.20,
        "متوازن":     0.30,
        "جريء":       0.45,
        "مغامر":      0.60,
    }
    active_fraction = kelly_map[risk_profile]

    st.markdown("---")
    st.subheader("🔬 فلاتر التحليل")
    min_ev_threshold = st.slider(
        "الحد الأدنى لـ EV٪", 0.0, 15.0, 3.0, 0.5
    ) / 100
    # ✅ إصلاح: الحد الأدنى للأودز 1.05 بدلاً من 1.25
    #    لتجنب استثناء الأودز المنخفضة ذات القيمة الحقيقية
    min_odds_filter = st.slider(
        "أودز أدنى مقبولة", 1.05, 2.50, 1.10, 0.05
    )
    max_odds_filter = st.slider(
        "أودز أعلى مقبولة", 2.0, 10.0, 5.0, 0.25
    )
    show_vig_chart  = st.checkbox("📊 عرض هوامش الشركات", value=True)
    show_ev_gauge   = st.checkbox("⚡ عرض مقياس EV",        value=True)
    max_matches     = st.slider("عدد المباريات المعروضة",  3, 20, 8)

# ══════════════════════════════════════════════════════
# الواجهة الرئيسية
# ══════════════════════════════════════════════════════
st.markdown("## 🕵️ آلة الدفع الذكية — Mastermind PRO")
st.markdown(
    f"نسخة **Elite** | منصة: **{target_bookmaker}** | "
    f"رأس المال: **{user_bankroll:,.0f}** | "
    f"مستوى المخاطرة: **{risk_profile}**"
)

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_button = st.button(
        "🚀 تشغيل الرادار الذكي",
        type="primary",
        use_container_width=True,
    )
with col_info:
    st.info(
        "⚡ يقارن النظام أودز شركتك مع إجماع السوق الحاد "
        "لاكتشاف أخطاء التسعير الحقيقية فقط."
    )

# ══════════════════════════════════════════════════════
# المعالجة الرئيسية
# ══════════════════════════════════════════════════════
if run_button:
    if not api_key_input:
        st.warning("⚠️ الرجاء إدخال API Key أولاً")
        st.stop()

    with st.spinner("🔄 جارٍ جلب البيانات وتحليل السوق..."):
        matches = fetch_odds(api_key_input, sport_key)

    if not matches:
        st.error(
            "❌ لم يتم جلب أي بيانات — "
            "تحقق من API Key والبطولة المختارة."
        )
        st.stop()

    st.success(f"✅ تم جلب **{len(matches)}** مباراة — جارٍ التحليل...")

    # ── ملخص السوق ──
    all_books: set = set()
    for m in matches:
        for b in m.get("bookmakers", []):
            all_books.add(b.get("title", ""))

    has_target = any(
        any(b.get("title") == target_bookmaker
            for b in m.get("bookmakers", []))
        for m in matches
    )
    sharp_found = any(
        any(b.get("title") in SHARP_BOOKS
            for b in m.get("bookmakers", []))
        for m in matches
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 مباريات",       len(matches))
    c2.metric("🏪 شركات",         len(all_books))
    c3.metric(
        f"🎯 {target_bookmaker}",
        "✅ متاحة" if has_target else "❌ غير متاحة",
    )
    c4.metric(
        "🔬 المرجع الحاد",
        "✅ Pinnacle" if sharp_found else "📊 متوسط السوق",
    )
    st.markdown("---")

    if not has_target:
        st.markdown(
            f'<div class="warning-box">'
            f'⚠️ <b>{target_bookmaker}</b> غير متاحة في هذه البطولة — '
            f'الاحتمالات الضمنية ستظهر صفراً. '
            f'اختر شركة أخرى من القائمة الجانبية.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════
    # حلقة تحليل المباريات
    # ══════════════════════════════════════════════════
    analyzed_count        = 0
    golden_opportunities: List[Dict[str, Any]] = []

    for idx, match in enumerate(matches):
        if analyzed_count >= max_matches:
            break

        home_team     = match.get("home_team", "Home")
        away_team     = match.get("away_team", "Away")
        commence_time = match.get("commence_time", "")
        bookmakers    = match.get("bookmakers", [])

        if not bookmakers:
            continue

        # ── إجماع السوق الحاد ──
        h2h_consensus = get_sharp_consensus(bookmakers, "h2h")
        if not h2h_consensus or len(h2h_consensus.get("true_probs", [])) < 2:
            continue

        outcomes_1x2   = h2h_consensus["outcomes"]
        true_probs_1x2 = h2h_consensus["true_probs"]

        h_true = d_true = a_true = 0.0
        for i, name in enumerate(outcomes_1x2):
            if name == home_team:
                h_true = true_probs_1x2[i]
            elif name == away_team:
                a_true = true_probs_1x2[i]
            else:
                d_true = true_probs_1x2[i]

        # Fallback إذا لم تتطابق الأسماء
        if h_true == 0.0 and a_true == 0.0 and len(true_probs_1x2) >= 2:
            h_true = true_probs_1x2[0]
            a_true = true_probs_1x2[-1]
            if len(true_probs_1x2) == 3:
                d_true = true_probs_1x2[1]

        # ── أودز الشركة المستهدفة H2H ──
        target_1x2 = get_target_book_odds(bookmakers, "h2h", target_bookmaker)
        h_odds_raw = target_1x2.get(home_team, 0.0)
        a_odds_raw = target_1x2.get(away_team, 0.0)

        h_implied = 1.0 / h_odds_raw if h_odds_raw > 1.0 else 0.0
        a_implied = 1.0 / a_odds_raw if a_odds_raw > 1.0 else 0.0
        draw_key  = next(
            (k for k in target_1x2 if k != home_team and k != away_team),
            None,
        )
        draw_odds_raw = target_1x2.get(draw_key, 0.0) if draw_key else 0.0
        d_implied     = 1.0 / draw_odds_raw if draw_odds_raw > 1.0 else 0.0

        # ── سوق الأهداف ──
        available_pts = get_available_totals_points(bookmakers)
        best_point    = (
            2.5 if 2.5 in available_pts
            else (available_pts[0] if available_pts else 2.5)
        )
        ou_consensus = get_sharp_consensus(bookmakers, "totals", best_point)
        target_ou    = get_target_book_odds(
            bookmakers, "totals", target_bookmaker, best_point
        )

        ou_over_true  = 0.50
        ou_under_true = 0.50
        if ou_consensus and len(ou_consensus.get("true_probs", [])) >= 2:
            for i, name in enumerate(ou_consensus["outcomes"]):
                if "over"  in name.lower():
                    ou_over_true  = ou_consensus["true_probs"][i]
                elif "under" in name.lower():
                    ou_under_true = ou_consensus["true_probs"][i]

        # ── بناء التوصيات ──
        final_recs = build_recommendations(
            home_team=home_team,         away_team=away_team,
            h_true=h_true,               a_true=a_true,
            h_odds_raw=h_odds_raw,       a_odds_raw=a_odds_raw,
            ou_over_true=ou_over_true,   ou_under_true=ou_under_true,
            target_ou=target_ou,         best_point=best_point,
            user_bankroll=user_bankroll, active_fraction=active_fraction,
            min_ev_threshold=min_ev_threshold,
            min_odds_filter=min_odds_filter,
            max_odds_filter=max_odds_filter,
        )

        for rec in final_recs:
            if rec.get("tier") == "golden":
                golden_opportunities.append(
                    {**rec, "home": home_team, "away": away_team}
                )

        # ── السرد الذكي ──
        # ✅ إصلاح: d_prob محذوف من الاستدعاء
        narrative = generate_narrative(
            home_team=home_team,
            away_team=away_team,
            h_prob=h_true,
            a_prob=a_true,
            ou_prob_over=ou_over_true,
            h2h_consensus=h2h_consensus,
            target_book=target_bookmaker,
            recs=final_recs,
        )

        # ── وقت المباراة ──
        time_str = "⏰ وقت غير محدد"
        if commence_time:
            try:
                dt = datetime.fromisoformat(
                    commence_time.replace("Z", "+00:00")
                )
                time_str = dt.strftime("⏰ %d/%m/%Y — %H:%M UTC")
            except (ValueError, AttributeError):
                pass

        # ✅ إصلاح: تمرير is_sharp لـ market_consensus_label
        book_quality = market_consensus_label(
            h2h_consensus.get("n_books", 0),
            h2h_consensus.get("vig", 10.0),
            h2h_consensus.get("is_sharp", False),
        )

        analyzed_count += 1

        # ══════════════════════════════════════════════
        # عرض بطاقة المباراة
        # ══════════════════════════════════════════════
        with st.container():
            st.markdown(f"""
            <div class="match-card">
                <div class="match-title">{home_team} ⚔️ {away_team}</div>
                <div class="match-subtitle">
                    {time_str} &nbsp;|&nbsp; {selected_sport_label}
                    &nbsp;|&nbsp; المرجع: {h2h_consensus.get('book_used', '—')}
                    &nbsp;|&nbsp; {book_quality}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_chart, col_gauge = st.columns([3, 2])
            with col_chart:
                render_prob_chart(
                    home_team, away_team,
                    h_true, a_true, d_true,
                    h_implied, a_implied, d_implied,
                    target_bookmaker,
                    chart_key=f"prob_{idx}",
                )
            with col_gauge:
                if final_recs and show_ev_gauge:
                    top_rec = max(final_recs, key=lambda x: x["ev"])
                    render_ev_gauge(
                        top_rec["ev"],
                        top_rec["confidence"],
                        gauge_key=f"gauge_{idx}",
                    )
                elif show_ev_gauge:
                    st.markdown(
                        '<div style="color:rgba(255,255,255,0.3);'
                        'text-align:center;padding:60px 0;font-size:13px;">'
                        '— لا توجد قيمة موجبة —</div>',
                        unsafe_allow_html=True,
                    )

            # مؤشرات سريعة
            st.markdown(f"""
            <div style="direction:rtl;">
                <div class="market-insight">
                    <span class="insight-label">🏠 احتمال فوز {home_team[:14]}</span>
                    <span class="insight-value">{h_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">🤝 احتمال التعادل</span>
                    <span class="insight-value">{d_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">✈️ احتمال فوز {away_team[:14]}</span>
                    <span class="insight-value">{a_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">📈 احتمال Over {best_point}</span>
                    <span class="insight-value">{ou_over_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">📉 احتمال Under {best_point}</span>
                    <span class="insight-value">{ou_under_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">
                        🔬 هامش {h2h_consensus.get('book_used', 'السوق')}
                    </span>
                    <span class="insight-value">
                        {h2h_consensus.get('vig', 0):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if show_vig_chart and len(bookmakers) >= 4:
                with st.expander("📊 هوامش شركات المراهنة", expanded=False):
                    render_vig_chart(bookmakers, chart_key=f"vig_{idx}")

            st.markdown(
                f'<div class="narrative-box">{narrative}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if final_recs:
                rec_cols = st.columns(len(final_recs))
                for i, rec in enumerate(final_recs):
                    with rec_cols[i]:
                        tier      = rec.get("tier", "extra")
                        css_class = {
                            "golden": "rec-golden",
                            "strong": "rec-strong",
                            "value":  "rec-value",
                        }.get(tier, "rec-extra")

                        ev_html = (
                            f'<span class="ev-badge-pos">'
                            f'EV: +{rec["ev"]*100:.1f}%</span>'
                            if rec["ev"] > 0
                            else
                            f'<span class="ev-badge-neg">'
                            f'EV: {rec["ev"]*100:.1f}%</span>'
                        )
                        kelly_pct = kelly_criterion(
                            rec["true_prob"], rec["odds"], active_fraction
                        ) * 100
                        fair_odds_str = (
                            f"{1 / rec['true_prob']:.2f}"
                            if rec["true_prob"] > 0
                            else "—"
                        )

                        st.markdown(f"""
                        <div class="{css_class}">
                            <div class="rec-label">{rec['label']}</div>
                            <div class="rec-bet-name">{rec['bet']}</div>
                            <div>
                                أودز {target_bookmaker}:
                                <span class="rec-odds-badge">
                                    {rec['odds']:.2f}
                                </span>
                            </div>
                            <div style="color:rgba(255,255,255,0.45);
                                        font-size:12px; margin:5px 0;">
                                الاحتمال الحقيقي: {rec['true_prob']*100:.1f}%
                                &nbsp;|&nbsp;
                                الأودز العادل: {fair_odds_str}
                            </div>
                            <div style="color:rgba(255,255,255,0.45);
                                        font-size:12px; margin-bottom:8px;">
                                الميزة: +{rec['edge_pct']:.1f}%
                            </div>
                            <div style="margin-top:10px;
                                        border-top:1px solid rgba(255,255,255,0.08);
                                        padding-top:10px;">
                                راهن بـ
                                <span class="rec-stake-highlight">
                                    {rec['stake']}
                                </span> وحدة
                            </div>
                            <div style="margin-top:7px;">{ev_html}</div>
                            <div style="color:rgba(255,255,255,0.28);
                                        font-size:11px; margin-top:5px;">
                                Confidence: {rec['confidence']}%
                                &nbsp;|&nbsp;
                                Kelly: {kelly_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="no-recs">'
                    '🚫 لا توجد قيمة موجبة في هذه المباراة — '
                    'الأمان أولاً، تجاهلها هو القرار الصحيح.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<hr style='border-color:rgba(255,215,0,0.08);margin:28px 0;'>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════
    # ملخص الفرص الذهبية
    # ══════════════════════════════════════════════════
    st.markdown("---")
    if golden_opportunities:
        st.markdown("## 💎 الفرص الذهبية المكتشفة اليوم")
        st.markdown(
            f"*اكتُشف **{len(golden_opportunities)}** خطأ تسعير حقيقي — "
            f"EV إيجابي مع ميزة واضحة على السوق*"
        )
        for opp in golden_opportunities:
            market_ar = "نتيجة" if opp.get("market") == "h2h" else "أهداف"
            st.markdown(f"""
            <div class="golden-opp-card">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;">
                    <div>
                        <span style="color:#FFD700;font-weight:900;font-size:15px;">
                            {opp.get('home','?')} vs {opp.get('away','?')}
                        </span>
                        <span style="color:rgba(255,255,255,0.35);
                                     font-size:11px;margin-right:8px;">
                            [{market_ar}]
                        </span><br>
                        <span style="color:rgba(255,255,255,0.75);font-size:13px;">
                            {opp['bet']}
                        </span>
                    </div>
                    <div style="text-align:center;">
                        <span style="color:#FFD700;font-size:22px;font-weight:900;">
                            {opp['odds']:.2f}
                        </span><br>
                        <span style="color:#00ff88;font-size:12px;font-weight:700;">
                            EV: +{opp['ev']*100:.1f}%
                        </span>
                    </div>
                    <div style="text-align:center;">
                        <span style="color:white;font-size:20px;font-weight:700;">
                            {opp['stake']}
                        </span><br>
                        <span style="color:rgba(255,255,255,0.35);font-size:11px;">
                            وحدة
                        </span>
                    </div>
                    <div style="text-align:center;">
                        <span style="color:rgba(255,255,255,0.6);font-size:12px;">
                            ميزة: +{opp['edge_pct']:.1f}%
                        </span><br>
                        <span style="color:rgba(255,255,255,0.35);font-size:11px;">
                            Conf: {opp['confidence']}%
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(
            "📭 لم يُكتشف أي خطأ تسعير بارز اليوم. "
            "هذا طبيعي — السوق الحاد يُسعّر بدقة معظم الوقت. "
            "حاول تقليل حد الـ EV أو تغيير البطولة."
        )

    if analyzed_count == 0:
        st.warning(
            "⚠️ لم يتم تحليل أي مباراة. "
            "تأكد من أن البطولة تحتوي على مباريات قادمة "
            f"وأن {target_bookmaker} تغطيها."
        )
    else:
        st.caption(
            f"✅ تم تحليل {analyzed_count} مباراة | "
            f"فرص ذهبية: {len(golden_opportunities)} | "
            f"الحد الأدنى للـ EV: {min_ev_threshold*100:.1f}%"
                )
