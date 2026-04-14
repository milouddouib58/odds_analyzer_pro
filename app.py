# -*- coding: utf-8 -*-
import streamlit as st
import requests
import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

st.set_page_config(page_title="🧠 Mastermind PRO", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;900&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; }
    body { background: #0a0a0f; }
    .main { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%); }
    .match-card {
        background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(20,20,35,0.98));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 215, 0, 0.15);
        box-shadow: 0 20px 60px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,215,0,0.1);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 50px;
        direction: rtl;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .match-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #FFD700, #FFA500, #FFD700, transparent);
    }
    .match-title {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .match-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 14px;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    .prob-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .rec-golden {
        background: linear-gradient(135deg, #1a1200, #2a1f00);
        border: 2px solid #FFD700;
        box-shadow: 0 0 30px rgba(255,215,0,0.4), inset 0 0 20px rgba(255,215,0,0.05);
        padding: 25px;
        border-radius: 18px;
        margin-top: 12px;
        color: white;
        position: relative;
        overflow: hidden;
        animation: goldenPulse 2.5s ease-in-out infinite;
    }
    .rec-golden::after {
        content: '⭐ VALUE BET';
        position: absolute;
        top: 12px;
        left: 12px;
        background: #FFD700;
        color: #000;
        font-size: 10px;
        font-weight: 900;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 1px;
    }
    @keyframes goldenPulse {
        0%, 100% { box-shadow: 0 0 25px rgba(255,215,0,0.3), inset 0 0 20px rgba(255,215,0,0.03); }
        50% { box-shadow: 0 0 50px rgba(255,215,0,0.6), inset 0 0 30px rgba(255,215,0,0.08); }
    }
    .rec-strong {
        background: linear-gradient(135deg, #001a0d, #002a15);
        border: 2px solid #00ff88;
        box-shadow: 0 0 25px rgba(0,255,136,0.2);
        padding: 25px;
        border-radius: 18px;
        margin-top: 12px;
        color: white;
    }
    .rec-value {
        background: linear-gradient(135deg, #0d0d1a, #12123a);
        border: 2px solid #4d79ff;
        box-shadow: 0 0 25px rgba(77,121,255,0.2);
        padding: 25px;
        border-radius: 18px;
        margin-top: 12px;
        color: white;
    }
    .rec-extra {
        background: linear-gradient(135deg, #0d1a1a, #0d2a2a);
        border: 2px solid #00bcd4;
        box-shadow: 0 0 20px rgba(0,188,212,0.15);
        padding: 25px;
        border-radius: 18px;
        margin-top: 12px;
        color: white;
    }
    .rec-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
        opacity: 0.85;
    }
    .rec-bet-name {
        font-size: 22px;
        font-weight: 900;
        margin: 12px 0;
        line-height: 1.3;
    }
    .rec-odds-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 20px;
        font-weight: 900;
        margin: 5px 0;
    }
    .rec-stake-highlight {
        font-size: 28px;
        font-weight: 900;
        color: #FFD700;
    }
    .ev-badge-pos {
        display: inline-block;
        background: linear-gradient(90deg, #00c851, #007e33);
        color: white;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 10px;
        margin-top: 8px;
    }
    .ev-badge-neg {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.5);
        font-size: 12px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 10px;
        margin-top: 8px;
    }
    .narrative-box {
        background: linear-gradient(135deg, rgba(255,215,0,0.05), rgba(255,140,0,0.03));
        border: 1px solid rgba(255,215,0,0.3);
        border-right: 5px solid #FFD700;
        padding: 25px 30px;
        border-radius: 16px;
        margin: 25px 0;
        font-size: 17px;
        line-height: 1.9;
        color: rgba(255,255,255,0.92);
        direction: rtl;
        position: relative;
    }
    .narrative-box::before {
        content: '🧠';
        position: absolute;
        top: -15px;
        right: 20px;
        font-size: 28px;
        background: #0d1117;
        padding: 0 5px;
    }
    .market-insight {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        direction: rtl;
    }
    .insight-label {
        color: rgba(255,255,255,0.5);
        font-size: 14px;
    }
    .insight-value {
        color: white;
        font-weight: 700;
        font-size: 16px;
    }
    .sidebar-header {
        background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,140,0,0.05));
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
        color: #FFD700;
        font-weight: 700;
        font-size: 16px;
    }
    .no-recs {
        background: rgba(255,50,50,0.08);
        border: 1px solid rgba(255,50,50,0.2);
        border-radius: 12px;
        padding: 15px 20px;
        color: rgba(255,150,150,0.8);
        font-size: 14px;
        text-align: center;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# الثوابت والإعدادات
# ──────────────────────────────────────────────────────
API_BASE_URL = "https://api.the-odds-api.com/v4"
MIN_STAKE = 22.0
MAX_STAKE_PCT = 0.15
MIN_STAKE_PCT = 0.01
SHARP_BOOKS = ["Pinnacle", "Betfair", "Matchbook", "SBOBet"]
SOFT_BOOKS_TARGET = ["1xBet", "Melbet", "Bet365", "William Hill", "Unibet"]

# ──────────────────────────────────────────────────────
# دوال جلب البيانات
# ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds(api_key: str, sport_key: str) -> list:
    """جلب الأودز مع cache لمدة 5 دقائق"""
    url = f"{API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk,us,au",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
        "includeLinks": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            remaining = resp.headers.get("x-requests-remaining", "?")
            st.session_state["api_remaining"] = remaining
            return resp.json()
        elif resp.status_code == 401:
            st.error("❌ API Key غير صالح أو منتهي الصلاحية")
        elif resp.status_code == 429:
            st.error("⚠️ تجاوزت حد الطلبات المسموح به لهذا الشهر")
        else:
            st.error(f"خطأ في الاتصال: {resp.status_code}")
    except requests.Timeout:
        st.error("⏱️ انتهت مهلة الاتصال. حاول مرة أخرى.")
    except Exception as e:
        st.error(f"خطأ غير متوقع: {str(e)}")
    return []

def fetch_available_sports(api_key: str) -> list:
    url = f"{API_BASE_URL}/sports"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

# ──────────────────────────────────────────────────────
# دوال الرياضيات والتحليل
# ──────────────────────────────────────────────────────
def remove_margin(odds_list: list) -> Optional[list]:
    """إزالة هامش البوكماكر واستخراج الاحتمالات الحقيقية"""
    if not odds_list or any(o <= 1.0 for o in odds_list):
        return None
    probs = [1.0 / o for o in odds_list]
    total = sum(probs)
    if total <= 0:
        return None
    return [p / total for p in probs]

def calculate_vig(odds_list: list) -> float:
    """احتساب هامش البوكماكر (Vig/Juice)"""
    if not odds_list:
        return 0.0
    probs = [1.0 / o for o in odds_list if o > 1.0]
    return (sum(probs) - 1.0) * 100

def calculate_ev(true_prob: float, odds: float) -> float:
    """حساب القيمة المتوقعة"""
    return (true_prob * (odds - 1.0)) - (1.0 - true_prob)

def kelly_criterion(true_prob: float, odds: float, fraction: float) -> float:
    """معادلة كيلي الكسرية"""
    b = odds - 1.0
    q = 1.0 - true_prob
    if b <= 0 or true_prob <= 0:
        return 0.0
    kelly = (b * true_prob - q) / b
    return max(0.0, kelly * fraction)

def calculate_stake(true_prob: float, odds: float, bankroll: float, fraction: float) -> float:
    """حساب المبلغ المقترح للرهان"""
    kelly_frac = kelly_criterion(true_prob, odds, fraction)
    if kelly_frac <= 0:
        return 0.0
    raw_stake = bankroll * kelly_frac
    capped = min(raw_stake, bankroll * MAX_STAKE_PCT)
    floored = max(capped, bankroll * MIN_STAKE_PCT)
    return max(floored, MIN_STAKE)

def get_sharp_consensus(bookmakers: list, market_key: str, point: Optional[float] = None) -> dict:
    """استخراج الإجماع الحاد من الكتب الذكية"""
    # أولاً: ابحث عن Sharp Books بالترتيب
    for book_name in SHARP_BOOKS:
        for b in bookmakers:
            if b.get("title", "").lower() == book_name.lower():
                for m in b.get("markets", []):
                    if m["key"] != market_key:
                        continue
                    outcomes = m.get("outcomes", [])
                    if market_key == "totals" and point is not None:
                        outcomes = [o for o in outcomes if abs(o.get("point", 0) - point) < 0.01]
                    if len(outcomes) >= 2:
                        odds_list = [o["price"] for o in outcomes]
                        true_probs = remove_margin(odds_list)
                        if true_probs:
                            return {
                                "outcomes": [o["name"] for o in outcomes],
                                "true_probs": true_probs,
                                "raw_odds": odds_list,
                                "vig": calculate_vig(odds_list),
                                "book_used": book_name,
                                "n_books": 1,
                                "odds_spread": (max(odds_list) - min(odds_list) if len(odds_list) >= 2 else 0.0),
                            }
    # ثانياً: متوسط مرجّح لكل الكتب المتاحة
    all_book_probs: Dict[str, list] = {}
    all_raw_odds: Dict[str, list] = {}
    count_books = 0
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] != market_key:
                continue
            outcomes = m.get("outcomes", [])
            if market_key == "totals" and point is not None:
                outcomes = [o for o in outcomes if abs(o.get("point", 0) - point) < 0.01]
            if len(outcomes) < 2:
                continue
            odds_list = [o["price"] for o in outcomes]
            probs = remove_margin(odds_list)
            if probs:
                count_books += 1
                for i, o in enumerate(outcomes):
                    key = o["name"]
                    all_book_probs.setdefault(key, []).append(probs[i])
                    all_raw_odds.setdefault(key, []).append(odds_list[i])
    if all_book_probs and count_books >= 2:
        names = list(all_book_probs.keys())
        avg_probs = [float(np.mean(all_book_probs[k])) for k in names]
        total = sum(avg_probs)
        norm_probs = [p / total for p in avg_probs] if total > 0 else avg_probs
        raw_odds_avg = [float(np.mean(all_raw_odds[k])) if all_raw_odds.get(k) else 2.0 for k in names]
        return {
            "outcomes": names,
            "true_probs": norm_probs,
            "raw_odds": raw_odds_avg,
            "vig": calculate_vig(raw_odds_avg),
            "book_used": f"إجماع {count_books} كتاب",
            "n_books": count_books,
            "odds_spread": (max(raw_odds_avg) - min(raw_odds_avg) if len(raw_odds_avg) >= 2 else 0.0),
        }
    return {}

def get_target_book_odds(bookmakers: list, market_key: str, target_book: str, point: Optional[float] = None) -> dict:
    """جلب أودز الشركة المحددة فقط"""
    for b in bookmakers:
        if b.get("title", "").lower() == target_book.lower():
            for m in b.get("markets", []):
                if m["key"] != market_key:
                    continue
                outcomes = m.get("outcomes", [])
                if market_key == "totals" and point is not None:
                    outcomes = [o for o in outcomes if abs(o.get("point", 0) - point) < 0.01]
                if outcomes:
                    return {o["name"]: o["price"] for o in outcomes}
    return {}

def detect_line_value(true_prob: float, book_odds: float, threshold: float = 0.03) -> dict:
    """كشف قيمة الخط"""
    ev = calculate_ev(true_prob, book_odds)
    implied_prob = 1.0 / book_odds if book_odds > 1.0 else 1.0
    edge = true_prob - implied_prob
    has_value = ev > threshold and edge > 0
    confidence = min(100, max(0, int(abs(ev) * 200 + edge * 150)))
    return {
        "has_value": has_value,
        "ev": ev,
        "edge_pct": edge * 100,
        "confidence": confidence,
        "implied_prob": implied_prob,
    }

def get_available_totals_points(bookmakers: list) -> list:
    """استخراج كل نقاط Over/Under المتاحة"""
    points: set = set()
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] == "totals":
                for o in m.get("outcomes", []):
                    if "point" in o:
                        points.add(o["point"])
    return sorted(list(points))

def calculate_market_consensus_strength(n_books: int, vig: float) -> str:
    """تقييم قوة الإجماع السوقي"""
    if n_books >= 8 and vig < 5:
        return "قوي جداً ✅"
    elif n_books >= 5 and vig < 8:
        return "جيد 🟡"
    elif n_books >= 3:
        return "مقبول ⚠️"
    else:
        return "ضعيف ❌"

def generate_ai_narrative(
    home_team: str, away_team: str,
    h_prob: float, a_prob: float, d_prob: float,
    ou_narrative: str, ou_prob_over: float,
    h2h_consensus: dict, ou_consensus: dict,
    target_book: str, recs: list
) -> str:
    """توليد سرد ذكي يقرأ ما بين السطور"""
    lines = []
    if h_prob > a_prob + 0.15:
        fav, underdog = home_team, away_team
        fav_prob, dog_prob = h_prob, a_prob
        fav_home = True
    elif a_prob > h_prob + 0.15:
        fav, underdog = away_team, home_team
        fav_prob, dog_prob = a_prob, h_prob
        fav_home = False
    else:
        fav = underdog = None
        fav_prob = dog_prob = 0.5
        fav_home = None

    is_balanced = abs(h_prob - a_prob) < 0.12
    is_one_sided = abs(h_prob - a_prob) > 0.30
    high_scoring_expected = ou_prob_over > 0.56
    low_scoring_expected = ou_prob_over < 0.44

    n_books = h2h_consensus.get("n_books", 0)
    vig = h2h_consensus.get("vig", 0)
    if n_books >= 7:
        lines.append(
            f"📊 **إجماع سوقي قوي:** {n_books} شركة مراهنة تُسعّر هذه المباراة "
            f"بهامش {vig:.1f}٪ — الأرقام موثوقة جداً."
        )

    if fav and is_one_sided:
        loc = "أمام جمهوره" if fav_home else "خارج أرضه"
        lines.append(
            f"⚡ **سيناريو الهيمنة:** {fav} يدخل هذه المباراة بتفوق لا لبس فيه "
            f"({fav_prob*100:.0f}٪ احتمال الفوز). "
            f"المنافس {underdog} يُعاني من ضعف في مواجهة الفرق القوية {loc}."
        )
        if high_scoring_expected:
            lines.append(
                f"🔥 **البيانات تتحدث:** احتمال عالٍ لمباراة بأهداف كثيرة — "
                f"السوق يُسعّر فوزاً ساحقاً لـ{fav}."
            )
        elif low_scoring_expected:
            lines.append(
                f"🔒 **الدفاع يحكم:** رغم الهيمنة، السوق يُشير لمباراة تكتيكية. "
                f"كسب نظيف (1-0 أو 2-0) هو السيناريو الأكثر منطقاً."
            )
    elif fav and not is_one_sided:
        lines.append(
            f"📌 **أفضلية معتدلة:** {fav} يتقدم بأفضلية ({fav_prob*100:.0f}٪) "
            f"لكنها ليست حاسمة. {underdog} لديه فرصة حقيقية ({dog_prob*100:.0f}٪)."
        )
    elif is_balanced:
        lines.append(
            "⚖️ **مباراة القدر المتساوي:** الأرقام تُشير لتعادل حقيقي في القوى. "
            "التركيز على الأهداف أذكى من الرهان على النتيجة."
        )

    if ou_prob_over > 0.60:
        lines.append(f"📈 **ضغط هجومي:** احتمال {ou_prob_over*100:.0f}٪ لتخطي 2.5 أهداف.")
    elif ou_prob_over < 0.40:
        lines.append(f"📉 **معركة دفاعية:** احتمال {(1-ou_prob_over)*100:.0f}٪ للبقاء تحت 2.5 أهداف.")

    golden_recs = [r for r in recs if r.get("tier") == "golden"]
    if golden_recs:
        for gr in golden_recs:
            true_odds_str = f"{1/gr['true_prob']:.2f}" if gr['true_prob'] > 0 else "—"
            lines.append(
                f"💎 **كسر التسعير اكتُشف!** {target_book} تُسعّر '{gr['bet']}' "
                f"بأودز {gr['odds']:.2f} بينما الاحتمال الحقيقي يستحق {true_odds_str}. "
                f"الميزة: **+{gr['ev']*100:.1f}٪**"
            )

    if is_balanced and not golden_recs:
        lines.append(
            "🚫 **تحذير المحترف:** مباراة متكافئة بلا قيمة واضحة = خطر عالٍ. "
            "ننصح بالتجاهل أو الرهان الرمزي فقط."
        )
    elif fav and is_one_sided and golden_recs:
        lines.append(
            f"🎯 **الخلاصة الذهبية:** تقاطع بين هيمنة {fav} وخطأ تسعير "
            f"{target_book} — فرصة نادرة."
        )
    elif fav and not golden_recs:
        lines.append(
            f"💡 **استراتيجية المحترف:** ابحث عن {fav} في الأكومولاتور "
            f"لتعظيم العائد مع الحفاظ على الأمان."
        )

    return "\n\n".join(lines) if lines else "🔍 البيانات غير كافية لتوليد قراءة تحليلية موثوقة."

# ──────────────────────────────────────────────────────
# دوال الرسم البياني
# ──────────────────────────────────────────────────────
def render_prob_comparison_chart(
    home_team: str, away_team: str,
    h_true: float, a_true: float, d_true: float,
    h_implied: float, a_implied: float, d_implied: float,
    target_book: str, index: int
):
    """مقارنة بين الاحتمال الحقيقي وما تعرضه الشركة"""
    categories = [home_team[-10:], "تعادل", away_team[-10:]]
    true_vals = [h_true * 100, d_true * 100, a_true * 100]
    implied_vals = [h_implied * 100, d_implied * 100, a_implied * 100]
    implied_vals = [max(0.0, v) for v in implied_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="الاحتمال الحقيقي للسوق", x=categories, y=true_vals,
        marker_color=["rgba(255,215,0,0.8)", "rgba(150,150,150,0.6)", "rgba(77,121,255,0.8)"],
        marker_line_color=["#FFD700", "#aaa", "#4d79ff"], marker_line_width=2,
        text=[f"{v:.1f}%" for v in true_vals], textposition="outside",
        textfont=dict(color="white", size=14, family="Tajawal"),
    ))
    fig.add_trace(go.Bar(
        name=f"ضمني {target_book}", x=categories, y=implied_vals,
        marker_color=["rgba(255,100,100,0.4)", "rgba(100,100,100,0.3)", "rgba(100,255,100,0.4)"],
        marker_line_color=["#ff6464", "#888", "#64ff64"], marker_line_width=1,
        text=[f"{v:.1f}%" for v in implied_vals], textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=11),
    ))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="white")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   ticksuffix="%", tickfont=dict(color="rgba(255,255,255,0.5)")),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"prob_compare_{index}")

def render_ev_gauge(ev: float, confidence: int, index: int, suffix: str = ""):
    """مقياس القيمة المتوقعة"""
    color = "#00ff88" if ev > 0 else "#ff4444"
    ev_pct = ev * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ev_pct,
        delta={"reference": 0, "valueformat": ".1f", "suffix": "%"},
        title={
            "text": (f"القيمة المتوقعة (EV)<br>"
                     f"<span style='font-size:12px;color:rgba(255,255,255,0.5)'>"
                     f"Confidence: {confidence}%</span>"),
            "font": {"size": 15, "color": "white", "family": "Tajawal"}
        },
        number={"suffix": "%", "font": {"size": 28, "color": color}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [-30, 30], "tickwidth": 1,
                     "tickcolor": "rgba(255,255,255,0.3)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 10}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0.3)",
            "borderwidth": 0,
            "steps": [
                {"range": [-30, -10], "color": "rgba(255,50,50,0.15)"},
                {"range": [-10, 0], "color": "rgba(255,165,0,0.10)"},
                {"range": [0, 10], "color": "rgba(0,255,100,0.10)"},
                {"range": [10, 30], "color": "rgba(0,255,100,0.25)"},
            ],
            "threshold": {"line": {"color": "#FFD700", "width": 3}, "thickness": 0.75, "value": 5},
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=15, r=15, t=50, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Tajawal"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ev_gauge_{index}_{suffix}")

def render_vig_radar(bookmakers_data: list, home_team: str, away_team: str, index: int):
    """رادار هوامش الشركات"""
    vigs: Dict[str, float] = {}
    for b in bookmakers_data:
        for m in b.get("markets", []):
            if m["key"] == "h2h":
                ods = [o["price"] for o in m.get("outcomes", []) if o["price"] > 1.0]
                if ods:
                    vigs[b["title"]] = calculate_vig(ods)
    if len(vigs) < 3:
        return
    sorted_vigs = sorted(vigs.items(), key=lambda x: x[1])[:12]
    names = [v[0] for v in sorted_vigs]
    values = [v[1] for v in sorted_vigs]
    colors = ["#00ff88" if v < 4 else "#FFD700" if v < 7 else "#ff6464" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors, marker_line_color="rgba(255,255,255,0.1)", marker_line_width=1,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        title=dict(
            text="🔬 مقارنة هوامش شركات المراهنة (كلما قلّ كان أفضل للاعب)",
            font=dict(size=13, color="rgba(255,255,255,0.7)", family="Tajawal")
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=max(250, len(names) * 28),
        margin=dict(l=10, r=50, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   ticksuffix="%", range=[0, max(values) * 1.3]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"vig_radar_{index}")

# ──────────────────────────────────────────────────────
# واجهة الشريط الجانبي
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-header">🧠 MASTERMIND PRO<br>'
        '<small style="font-weight:300;opacity:0.7">نظام تحليل الأودز المتقدم</small></div>',
        unsafe_allow_html=True
    )

    api_key_input = st.text_input(
        "🔑 Odds API Key", type="password", placeholder="أدخل مفتاح API الخاص بك"
    )
    if "api_remaining" in st.session_state:
        st.caption(f"📡 طلبات متبقية هذا الشهر: **{st.session_state['api_remaining']}**")

    st.markdown("---")
    target_bookmaker = st.radio(
        "🎯 منصة الرهان المستهدفة",
        ["1xBet", "Melbet", "Bet365", "Unibet", "William Hill"],
        help="الشركة التي ستراهن عليها فعلياً"
    )
    st.markdown("---")

    SPORTS_MAP = {
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 الدوري الإنجليزي": "soccer_epl",
        "🇪🇸 الدوري الإسباني": "soccer_spain_la_liga",
        "🇩🇪 الدوري الألماني": "soccer_germany_bundesliga",
        "🇮🇹 الدوري الإيطالي": "soccer_italy_serie_a",
        "🇫🇷 الدوري الفرنسي": "soccer_france_ligue_one",
        "🏆 دوري أبطال أوروبا": "soccer_uefa_champs_league",
        "🌍 الدوري الأوروبي": "soccer_uefa_europa_league",
        "🏀 NBA كرة السلة": "basketball_nba",
        "🎾 تنس ATP": "tennis_atp",
    }
    selected_sport_label = st.selectbox("🏆 اختر البطولة", list(SPORTS_MAP.keys()))
    sport_key = SPORTS_MAP[selected_sport_label]

    st.markdown("---")
    st.subheader("💰 إدارة الرأسمال")
    user_bankroll = st.number_input(
        "💵 الرصيد الحالي (وحدة)", min_value=100.0, max_value=1_000_000.0,
        value=1000.0, step=100.0, help="إجمالي رصيدك المخصص للمراهنة"
    )
    risk_profile = st.select_slider(
        "📊 مستوى المخاطرة",
        options=["محافظ جداً", "محافظ", "متوازن", "جريء", "مغامر"],
        value="متوازن"
    )
    kelly_map = {
        "محافظ جداً": 0.10, "محافظ": 0.20, "متوازن": 0.30,
        "جريء": 0.45, "مغامر": 0.60
    }
    active_fraction = kelly_map[risk_profile]

    st.markdown("---")
    st.subheader("🔬 فلاتر التحليل")
    min_ev_threshold = st.slider("الحد الأدنى لـ EV%", 0.0, 15.0, 3.0, 0.5) / 100
    min_odds_filter = st.slider("أودز أدنى مقبولة", 1.10, 2.50, 1.25, 0.05)
    max_odds_filter = st.slider("أودز أعلى مقبولة", 2.0, 10.0, 4.0, 0.25)
    show_vig_chart = st.checkbox("📊 عرض رادار هوامش الشركات", value=True)
    show_ev_gauge = st.checkbox("⚡ عرض مقياس EV", value=True)
    max_matches_to_show = st.slider("عدد المباريات المعروضة", 3, 20, 8)

# ──────────────────────────────────────────────────────
# الواجهة الرئيسية
# ──────────────────────────────────────────────────────
st.markdown("## 🕵️ آلة الدفع الذكية — Mastermind PRO")
st.markdown(
    f"نسخة **Elite** | منصة الرهان: **{target_bookmaker}** | "
    f"رأس المال: **{user_bankroll:,.0f}** وحدة | "
    f"مستوى المخاطرة: **{risk_profile}**"
)

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_button = st.button("🚀 تشغيل الرادار الذكي", type="primary", use_container_width=True)
with col_info:
    st.info(
        "⚡ يقوم النظام بمقارنة أودز شركتك مع إجماع السوق الحاد "
        "(Pinnacle + متوسط السوق) لاكتشاف أخطاء التسعير."
    )

# ──────────────────────────────────────────────────────
# المعالجة الرئيسية
# ──────────────────────────────────────────────────────
if run_button:
    if not api_key_input:
        st.warning("⚠️ الرجاء إدخال API Key أولاً")
        st.stop()

    with st.spinner("🔄 جارٍ تحليل السوق..."):
        matches = fetch_odds(api_key_input, sport_key)

    if not matches:
        st.error("❌ لم يتم جلب أي بيانات.")
        st.stop()

    st.success(f"✅ تم جلب **{len(matches)}** مباراة — جارٍ التحليل العميق...")

    # ملخص السوق
    total_books_found: set = set()
    for m in matches:
        for b in m.get("bookmakers", []):
            total_books_found.add(b.get("title", ""))
    has_target = any(
        any(b.get("title") == target_bookmaker for b in m.get("bookmakers", []))
        for m in matches
    )

    cols_summary = st.columns(4)
    with cols_summary[0]:
        st.metric("📅 مباريات متاحة", len(matches))
    with cols_summary[1]:
        st.metric("🏪 شركات مراهنة", len(total_books_found))
    with cols_summary[2]:
        st.metric(f"🎯 {target_bookmaker}", f"{'✅ متاحة' if has_target else '❌ غير متاحة'}")
    with cols_summary[3]:
        sharp_found = any(
            any(b.get("title") in SHARP_BOOKS for b in m.get("bookmakers", []))
            for m in matches
        )
        st.metric("🔬 مرجع حاد (Sharp)", "✅ Pinnacle" if sharp_found else "📊 متوسط السوق")
    st.markdown("---")

    # ──────────────────────────────────────────────────
    # حلقة تحليل المباريات
    # ──────────────────────────────────────────────────
    analyzed_count = 0
    golden_opportunities: list = []

    for index, match in enumerate(matches):
        if analyzed_count >= max_matches_to_show:
            break

        home_team = match.get("home_team", "Home")
        away_team = match.get("away_team", "Away")
        commence_time = match.get("commence_time", "")
        bookmakers = match.get("bookmakers", [])

        if not bookmakers:
            continue

        # ── إجماع السوق الحاد ──
        h2h_consensus = get_sharp_consensus(bookmakers, "h2h")
        if not h2h_consensus or len(h2h_consensus.get("true_probs", [])) < 2:
            continue

        outcomes_1x2 = h2h_consensus["outcomes"]
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
        if h_true == 0.0 and a_true == 0.0:
            if len(true_probs_1x2) >= 2:
                h_true = true_probs_1x2[0]
                a_true = true_probs_1x2[-1]
                if len(true_probs_1x2) == 3:
                    d_true = true_probs_1x2[1]

        # ── أودز الشركة المستهدفة ──
        target_1x2 = get_target_book_odds(bookmakers, "h2h", target_bookmaker)

        # ── حساب الاحتمالات الضمنية بأمان ──
        h_odds_raw = target_1x2.get(home_team, 0)
        a_odds_raw = target_1x2.get(away_team, 0)
        h_implied = 1.0 / h_odds_raw if h_odds_raw > 1.0 else 0.0
        a_implied = 1.0 / a_odds_raw if a_odds_raw > 1.0 else 0.0

        # احتمال ضمني للتعادل من الشركة المستهدفة
        draw_key = next((k for k in target_1x2 if k != home_team and k != away_team), None)
        draw_odds_raw = target_1x2.get(draw_key, 0) if draw_key else 0
        d_implied = 1.0 / draw_odds_raw if draw_odds_raw > 1.0 else 0.0

        # ── سوق الأهداف ──
        available_points = get_available_totals_points(bookmakers)
        best_point = 2.5 if 2.5 in available_points else (available_points[0] if available_points else 2.5)
        ou_consensus = get_sharp_consensus(bookmakers, "totals", best_point)
        target_ou = get_target_book_odds(bookmakers, "totals", target_bookmaker, best_point)

        ou_over_true = 0.50
        ou_under_true = 0.50
        if ou_consensus and len(ou_consensus.get("true_probs", [])) >= 2:
            for i, name in enumerate(ou_consensus["outcomes"]):
                if "over" in name.lower():
                    ou_over_true = ou_consensus["true_probs"][i]
                elif "under" in name.lower():
                    ou_under_true = ou_consensus["true_probs"][i]

        # ── بناء التوصيات ──
        recs: list = []

        # H2H
        for team_name, true_prob, odds_raw in [
            (home_team, h_true, h_odds_raw),
            (away_team, a_true, a_odds_raw),
        ]:
            if odds_raw <= 1.0:
                continue
            book_odds = odds_raw
            if not (min_odds_filter <= book_odds <= max_odds_filter):
                continue

            value_info = detect_line_value(true_prob, book_odds, threshold=min_ev_threshold)
            stake = max(
                round(calculate_stake(true_prob, book_odds, user_bankroll, active_fraction)),
                int(MIN_STAKE)
            )

            rec_base: Dict[str, Any] = {
                "bet": f"فوز {team_name}",
                "odds": book_odds,
                "true_prob": true_prob,
                "implied_prob": value_info["implied_prob"],
                "stake": stake,
                "ev": value_info["ev"],
                "edge_pct": value_info["edge_pct"],
                "confidence": value_info["confidence"],
                "market": "h2h",
            }

            if value_info["has_value"] and value_info["ev"] > min_ev_threshold:
                rec_base["label"] = "💎 خطأ تسعير مكتشف!"
                rec_base["tier"] = "golden"
                rec_base["stake"] = max(
                    round(min(stake * 1.25, user_bankroll * MAX_STAKE_PCT)),
                    int(MIN_STAKE)
                )
                recs.append(rec_base)
                golden_opportunities.append({**rec_base, "home": home_team, "away": away_team})
            elif (not value_info["has_value"] and book_odds < 1.65 and true_prob > 0.55):
                rec_base["label"] = "🟢 مفضل ضاغط"
                rec_base["tier"] = "strong"
                recs.append(rec_base)
            elif (not value_info["has_value"] and 1.80 <= book_odds <= max_odds_filter and value_info["ev"] > -0.08):
                rec_base["label"] = "🔵 قيمة مقبولة"
                rec_base["tier"] = "value"
                recs.append(rec_base)

        # O/U - بحث آمن عن المفاتيح
        if ou_consensus and target_ou:
            over_key = next((k for k in target_ou if "over" in k.lower()), None)
            under_key = next((k for k in target_ou if "under" in k.lower()), None)

            for ou_key, ou_true_p, ou_label_ar in [
                (over_key, ou_over_true, f"أكثر من {best_point} أهداف"),
                (under_key, ou_under_true, f"أقل من {best_point} أهداف"),
            ]:
                if not ou_key:
                    continue
                book_ou_odds = target_ou.get(ou_key, 0)
                if book_ou_odds <= 1.0:
                    continue
                if not (min_odds_filter <= book_ou_odds <= max_odds_filter):
                    continue

                ou_value = detect_line_value(ou_true_p, book_ou_odds, threshold=min_ev_threshold)
                ou_stake = max(
                    round(calculate_stake(ou_true_p, book_ou_odds, user_bankroll, active_fraction)),
                    int(MIN_STAKE)
                )

                ou_rec: Dict[str, Any] = {
                    "bet": ou_label_ar,
                    "odds": book_ou_odds,
                    "true_prob": ou_true_p,
                    "implied_prob": ou_value["implied_prob"],
                    "stake": ou_stake,
                    "ev": ou_value["ev"],
                    "edge_pct": ou_value["edge_pct"],
                    "confidence": ou_value["confidence"],
                    "market": "totals",
                }

                if ou_value["has_value"] and ou_value["ev"] > min_ev_threshold:
                    ou_rec["label"] = "💎 قيمة في الأهداف!"
                    ou_rec["tier"] = "golden"
                    ou_rec["stake"] = max(round(ou_stake * 1.2), int(MIN_STAKE))
                    recs.append(ou_rec)
                    golden_opportunities.append({**ou_rec, "home": home_team, "away": away_team})
                elif (1.30 <= book_ou_odds <= 1.85 and ou_value["ev"] > -0.06):
                    ou_rec["label"] = "🟡 أهداف مثيرة للاهتمام"
                    ou_rec["tier"] = "extra"
                    recs.append(ou_rec)

        # ── ترتيب التوصيات ──
        tier_order = {"golden": 0, "strong": 1, "value": 2, "extra": 3}
        recs_sorted = sorted(
            recs, key=lambda x: (-x["ev"], tier_order.get(x.get("tier", "extra"), 4))
        )
        final_recs = recs_sorted[:4]

        # ── السرد الذكي ──
        ou_narrative_label = "مفتوح" if ou_over_true > 0.5 else "مغلق"
        narrative = generate_ai_narrative(
            home_team, away_team, h_true, a_true, d_true,
            ou_narrative_label, ou_over_true,
            h2h_consensus, ou_consensus if ou_consensus else {},
            target_bookmaker, final_recs
        )

        # ── عرض البطاقة ──
        analyzed_count += 1

        time_str = ""
        if commence_time:
            try:
                dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                time_str = dt.strftime("⏰ %d/%m/%Y — %H:%M UTC")
            except (ValueError, AttributeError):
                time_str = "⏰ وقت غير محدد"

        book_quality = calculate_market_consensus_strength(
            h2h_consensus.get("n_books", 0), h2h_consensus.get("vig", 10)
        )

        with st.container():
            st.markdown(f"""
            <div class="match-card">
                <div class="match-title">{home_team} ⚔️ {away_team}</div>
                <div class="match-subtitle">
                    {time_str} &nbsp;|&nbsp; {selected_sport_label} &nbsp;|&nbsp;
                    المرجع: {h2h_consensus.get('book_used','—')} &nbsp;|&nbsp;
                    إجماع السوق: {book_quality}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_chart, col_gauges = st.columns([3, 2])
            with col_chart:
                render_prob_comparison_chart(
                    home_team, away_team, h_true, a_true, d_true,
                    h_implied, a_implied, d_implied, target_bookmaker, index
                )
            with col_gauges:
                top_ev_rec = max(final_recs, key=lambda x: x["ev"]) if final_recs else None
                if top_ev_rec and show_ev_gauge:
                    render_ev_gauge(top_ev_rec["ev"], top_ev_rec["confidence"], index, "top")

            # مؤشرات سريعة
            st.markdown(f"""
            <div style="direction:rtl;">
                <div class="market-insight">
                    <span class="insight-label">🏠 احتمال فوز {home_team[:12]}</span>
                    <span class="insight-value">{h_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">🤝 احتمال التعادل</span>
                    <span class="insight-value">{d_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">✈️ احتمال فوز {away_team[:12]}</span>
                    <span class="insight-value">{a_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">📈 احتمال +{best_point} أهداف</span>
                    <span class="insight-value">{ou_over_true*100:.1f}%</span>
                </div>
                <div class="market-insight">
                    <span class="insight-label">🔬 هامش {h2h_consensus.get('book_used','السوق')}</span>
                    <span class="insight-value">{h2h_consensus.get('vig',0):.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if show_vig_chart and len(bookmakers) >= 4:
                with st.expander("📊 رادار هوامش شركات المراهنة", expanded=False):
                    render_vig_radar(bookmakers, home_team, away_team, index)

            if narrative:
                st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

            if final_recs:
                st.markdown("<br>", unsafe_allow_html=True)
                n_cols = min(len(final_recs), 4)
                rec_cols = st.columns(n_cols)
                for i, rec in enumerate(final_recs):
                    with rec_cols[i % n_cols]:
                        tier = rec.get("tier", "extra")
                        css_class = {
                            "golden": "rec-golden",
                            "strong": "rec-strong",
                            "value": "rec-value",
                        }.get(tier, "rec-extra")

                        ev_html = (
                            f'<span class="ev-badge-pos">EV: +{rec["ev"]*100:.1f}%</span>'
                            if rec["ev"] > 0 else
                            f'<span class="ev-badge-neg">EV: {rec["ev"]*100:.1f}%</span>'
                        )
                        edge_text = f"ميزة: {rec['edge_pct']:.1f}%" if rec.get("edge_pct") else ""
                        kelly_pct = kelly_criterion(rec["true_prob"], rec["odds"], active_fraction) * 100

                        st.markdown(f"""
                        <div class="{css_class}">
                            <div class="rec-label">{rec['label']}</div>
                            <div class="rec-bet-name">{rec['bet']}</div>
                            <div>
                                أودز {target_bookmaker}:
                                <span class="rec-odds-badge">{rec['odds']:.2f}</span>
                            </div>
                            <div style="color:rgba(255,255,255,0.5); font-size:12px; margin:6px 0;">
                                الاحتمال الحقيقي: {rec['true_prob']*100:.1f}% &nbsp;|&nbsp; {edge_text}
                            </div>
                            <div style="margin-top:12px; border-top:1px solid rgba(255,255,255,0.1); padding-top:12px;">
                                راهن بـ <span class="rec-stake-highlight">{rec['stake']}</span> وحدة
                            </div>
                            <div style="margin-top:8px;">{ev_html}</div>
                            <div style="color:rgba(255,255,255,0.3); font-size:11px; margin-top:6px;">
                                Confidence: {rec['confidence']}% | Kelly%: {kelly_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="no-recs">'
                    '🚫 لا توجد توصيات تتجاوز معايير الجودة — الأمان أولاً'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                "<br><hr style='border-color:rgba(255,215,0,0.1);margin:30px 0;'>",
                unsafe_allow_html=True
            )

    # ──────────────────────────────────────────────────
    # ملخص الفرص الذهبية
    # ──────────────────────────────────────────────────
    if golden_opportunities:
        st.markdown("---")
        st.markdown("## 💎 ملخص الفرص الذهبية المكتشفة")
        st.markdown(f"*تم اكتشاف **{len(golden_opportunities)}** خطأ تسعير في السوق اليوم*")
        for opp in golden_opportunities:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(255,215,0,0.08), rgba(255,140,0,0.04));
                border: 1px solid rgba(255,215,0,0.3);
                border-radius: 12px;
                padding: 15px 20px;
                margin: 8px 0;
                direction: rtl;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <span style="color:#FFD700; font-weight:900; font-size:16px;">
                        {opp.get('home','?')} vs {opp.get('away','?')}
                    </span><br>
                    <span style="color:rgba(255,255,255,0.7); font-size:14px;">
                        {opp['bet']}
                    </span>
                </div>
                <div style="text-align:left;">
                    <span style="color:#FFD700; font-size:20px; font-weight:900;">
                        {opp['odds']:.2f}
                    </span><br>
                    <span style="color:#00ff88; font-size:13px;">
                        EV: +{opp['ev']*100:.1f}%
                    </span>
                </div>
                <div style="text-align:center;">
                    <span style="color:white; font-size:18px; font-weight:700;">
                        {opp['stake']} وحدة
                    </span><br>
                    <span style="color:rgba(255,255,255,0.4); font-size:12px;">
                        حجم الرهان
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif analyzed_count > 0:
        st.info(
            "📭 لم يُكتشف أي خطأ تسعير بارز في هذا التوقيت. "
            "حاول تقليل حد الـ EV في الفلاتر أو انتظر تحديث الأودز."
        )

    if analyzed_count == 0:
        st.warning(
            "⚠️ لم يتم تحليل أي مباراة. "
            "تأكد من أن البطولة المختارة تحتوي على مباريات قادمة، "
            f"وأن {target_bookmaker} تُغطي هذه البطولة."
)
