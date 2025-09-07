# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st

# --- استيراد الدوال ---
try:
    from odds_math import (
        poisson_prediction,
        aggregate_prices,
        implied_from_decimal,
        shin_fair_probs,
        kelly_suggestions,
    )
    from gemini_helper import analyze_with_gemini
    import odds_provider_theoddsapi as odds_api
except ImportError as e:
    st.error(f"خطأ في الاستيراد: {e}. تأكد من وجود كل الملفات المساعدة!")
    st.stop()

# --- إعدادات الصفحة والـ CSS ---
st.set_page_config(page_title="Odds Strategist PRO", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .prob-bar-container{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
    .prob-bar-title{display:flex;justify-content:space-between;font-size:.9em;color:#b0b8c2}
    .prob-bar{width:100%;background-color:#334155;border-radius:5px;overflow:hidden;height:15px}
    .prob-bar-fill{height:100%;border-radius:5px;transition:width .5s ease-in-out;text-align:center;color:#fff;font-size:.8em;font-weight:700;line-height:15px}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_prob_bar(label, probability, color):
    p = 0.0 if probability is None else float(probability)
    p = max(0.0, min(1.0, p))  # clamp 0..1
    pct = p * 100.0
    return f"""
    <div class="prob-bar-container">
      <div class="prob-bar-title"><span>{label}</span><span>{pct:.1f}%</span></div>
      <div class="prob-bar"><div class="prob-bar-fill" style="width:{pct}%;background-color:{color};"></div></div>
    </div>
    """

# --- عناوين ---
st.markdown("<h1>Odds Strategist PRO 🧠</h1>", unsafe_allow_html=True)
st.markdown("### التحليل المزدوج: استراتيجيات السوق + التوقعات الإحصائية (بواسون)")

# --- كاش للطلبات ---
@st.cache_data(ttl=300)
def cached_list_sports():
    return odds_api.list_sports()

@st.cache_data(ttl=90)
def cached_fetch_odds_for_sport(sport_key: str, regions: str, markets: str):
    return odds_api.fetch_odds_for_sport(sport_key, regions, markets)

# --- الشريط الجانبي ---
def load_api_keys():
    st.sidebar.header("🔑 إعدادات المفاتيح")
    odds_key, gemini_key = None, None
    if 'ODDS_API_KEY' in st.secrets:
        odds_key = st.secrets['ODDS_API_KEY']
        st.sidebar.success("✅ تم تحميل مفتاح The Odds API.")
    else:
        odds_key = st.sidebar.text_input("The Odds API Key", type="password")
    if 'GEMINI_API_KEY' in st.secrets:
        gemini_key = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("✅ تم تحميل مفتاح Gemini API.")
    else:
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    if odds_key:
        os.environ["ODDS_API_KEY"] = odds_key
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    return odds_key, gemini_key

odds_api_key, gemini_api_key = load_api_keys()

st.sidebar.header("🏦 إدارة المحفظة")
bankroll = st.sidebar.number_input("حجم المحفظة ($)", min_value=1.0, value=100.0, step=10.0)
kelly_scale = st.sidebar.slider("معامل كيلي (Kelly Scale)", 0.05, 1.0, 0.25, 0.05)

st.sidebar.header("📊 إحصائيات بواسون (أداء الفرق)")
st.sidebar.info("أدخل متوسط أداء الفرق لكل مباراة. ملاحظة: قيمة دفاع أعلى تقلل الأهداف المتوقعة على المنافس.")
home_attack = st.sidebar.number_input("قوة هجوم الفريق المضيف", min_value=0.0, value=1.5, step=0.1)
home_defense = st.sidebar.number_input("قوة دفاع الفريق المضيف", min_value=0.0, value=1.0, step=0.1)
away_attack = st.sidebar.number_input("قوة هجوم الفريق الضيف", min_value=0.0, value=1.2, step=0.1)
away_defense = st.sidebar.number_input("قوة دفاع الفريق الضيف", min_value=0.0, value=1.3, step=0.1)
home_adv = st.sidebar.slider("أفضلية الملعب (Home Advantage)", min_value=1.00, max_value=1.30, value=1.10, step=0.01)

st.sidebar.header("⚙️ إعدادات السوق")
if not odds_api_key:
    st.sidebar.warning("الرجاء إدخال مفتاح The Odds API.")
    st.stop()

try:
    sports = cached_list_sports()
    sport_options = {f"{s.get('group')} - {s.get('title')}": s.get("key") for s in sports}
    selected_sport_label = st.sidebar.selectbox("اختر الرياضة:", list(sport_options.keys()))
    sport_key = sport_options[selected_sport_label]
    regions = st.sidebar.multiselect("المناطق:", ["eu", "uk", "us", "au"], default=["eu", "uk"])
    markets = st.sidebar.multiselect("الأسواق:", ["h2h", "totals"], default=["h2h", "totals"])
except Exception as e:
    st.error(f"لا يمكن جلب الرياضات. تأكد من صحة مفتاح The Odds API. الخطأ: {e}")
    st.stop()

# --- جلب البيانات ---
if st.button("🚀 جلب وتحليل المباريات"):
    with st.spinner(f"جاري جلب مباريات {selected_sport_label}..."):
        try:
            events, meta = cached_fetch_odds_for_sport(sport_key, ",".join(regions), ",".join(markets))
            st.session_state["events_data"] = {"events": events, "meta": dict(meta)}
            st.success(f"تم جلب {len(events)} مباراة.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء جلب أسعار المباريات: {e}")

# --- عرض وتحليل المباريات ---
if "events_data" in st.session_state:
    events = st.session_state["events_data"]["events"]
    match_options = {f"{ev.get('home_team')} vs {ev.get('away_team')}": i for i, ev in enumerate(events)}

    if not match_options:
        st.info("لا توجد مباريات متاحة حالياً لهذه الإعدادات.")
    else:
        selected_match_label = st.selectbox("اختر مباراة من القائمة:", list(match_options.keys()))
        event = events[match_options[selected_match_label]]
        home_team_name = event['home_team']
        away_team_name = event['away_team']

        tab1, tab2, tab3, tab4 = st.tabs(["📊 التحليل المزدوج", "📈 تفاصيل 1x2", "⚽️ تفاصيل الأهداف", "🤖 استشارة Gemini"])

        # 1) تحليل السوق H2H (ديناميكي 2-طريق/3-طريق)
        h2h_prices = odds_api.extract_h2h_prices(event)
        present_outcomes = [k for k, v in h2h_prices.items() if v]  # outcomes المتاحة فعلاً
        agg_odds_h2h, fair_h2h, sugg_h2h = {}, {}, {}
        if present_outcomes:
            agg_odds_h2h = {s: aggregate_prices(h2h_prices[s], 'best') for s in present_outcomes}
            fair_h2h = shin_fair_probs(implied_from_decimal(agg_odds_h2h))
            # اقتراحات كيلي مع سقف حماية
            sugg_h2h = kelly_suggestions(fair_h2h, agg_odds_h2h, bankroll, kelly_scale, max_fraction=0.25)

        # 2) التحليل الإحصائي (بواسون)
        poisson_probs = poisson_prediction(home_attack, home_defense, away_attack, away_defense, home_adv=home_adv)

        color_map = {"home": "#4a90e2", "draw": "#f5a623", "away": "#e24a4a"}
        def side_label(side: str):
            if side == 'home':
                return home_team_name
            if side == 'away':
                return away_team_name
            return "التعادل"

        with tab1:
            st.header("مقارنة بين رأي السوق ورأي الإحصاء")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("تحليل السوق (Fair Odds)")
                if fair_h2h:
                    for side in present_outcomes:
                        st.markdown(
                            render_prob_bar(side_label(side), fair_h2h.get(side, 0), color_map.get(side, "#888")),
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("لا توجد بيانات سوقية لنتيجة المباراة (H2H) لهذه المباراة.")
            with col2:
                st.subheader("التحليل الإحصائي (Poisson)")
                stats_outcomes = ['home', 'draw', 'away'] if 'draw' in present_outcomes else ['home', 'away']
                for side in stats_outcomes:
                    st.markdown(
                        render_prob_bar(side_label(side), poisson_probs.get(side, 0), color_map.get(side, "#888")),
                        unsafe_allow_html=True,
                    )

        with tab2:
            st.header("تفاصيل سوق نتيجة المباراة (1x2)")
            if not sugg_h2h or not any(s.get('edge', 0) > 0 for s in sugg_h2h.values()):
                st.info("لا توجد فرص قيمة (Value) واضحة في هذا السوق.")
            else:
                for side, suggestion in sugg_h2h.items():
                    if suggestion.get('edge', 0) > 0:
                        with st.container():
                            st.subheader(f"🎯 فرصة قيمة: {side_label(side)}")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("أفضل سعر", f"{agg_odds_h2h.get(side, 0):.2f}")
                            c2.metric("الأفضلية (Edge)", f"+{suggestion['edge']*100:.2f}%")
                            c3.metric("الرهان المقترح", f"${suggestion['stake_amount']:.2f}")

        with tab3:
            st.header("تفاصيل سوق الأهداف (Over/Under)")
            totals_lines = odds_api.extract_totals_lines(event)
            if not totals_lines:
                st.info("لا توجد بيانات لسوق الأهداف لهذه المباراة.")
            else:
                def _is_float(x):
                    try:
                        float(x)
                        return True
                    except Exception:
                        return False
                numeric_keys = sorted([k for k in totals_lines.keys() if _is_float(k)], key=lambda x: float(x))
                if not numeric_keys:
                    st.info("لا توجد خطوط صالحة للأهداف.")
                else:
                    selected_line = st.selectbox("اختر خط الأهداف:", numeric_keys)
                    line_data = totals_lines[selected_line]
                    agg_odds_ou = {
                        'over': aggregate_prices(line_data.get('over', []), 'best'),
                        'under': aggregate_prices(line_data.get('under', []), 'best'),
                    }
                    if agg_odds_ou.get('over', 0) > 0 and agg_odds_ou.get('under', 0) > 0:
                        imps_ou = implied_from_decimal(agg_odds_ou)
                        fair_ou = shin_fair_probs(imps_ou)
                        sugg_ou = kelly_suggestions(fair_ou, agg_odds_ou, bankroll, kelly_scale, max_fraction=0.25)

                        st.subheader(f"الاحتمالات العادلة لخط {selected_line}")
                        st.markdown(render_prob_bar(f"Over {selected_line}", fair_ou.get('over', 0), '#22c55e'), unsafe_allow_html=True)
                        st.markdown(render_prob_bar(f"Under {selected_line}", fair_ou.get('under', 0), '#ef4444'), unsafe_allow_html=True)

                        if any(s.get('edge', 0) > 0 for s in sugg_ou.values()):
                            for side, suggestion in sugg_ou.items():
                                if suggestion.get('edge', 0) > 0:
                                    with st.container():
                                        st.subheader(f"🎯 فرصة قيمة: {side.capitalize()} {selected_line}")
                                        c1, c2, c3 = st.columns(3)
                                        c1.metric("أفضل سعر", f"{agg_odds_ou.get(side, 0):.2f}")
                                        c2.metric("الأفضلية (Edge)", f"+{suggestion['edge']*100:.2f}%")
                                        c3.metric("الرهان المقترح", f"${suggestion['stake_amount']:.2f}")
                        else:
                            st.info("لا توجد فرص قيمة واضحة في هذا الخط.")
                    else:
                        st.warning("لا توجد أسعار كافية لتحليل هذا الخط.")

        with tab4:
            st.header("اطلب استشارة من 'الاستراتيجي'")
            if st.button("حلل يا استراتيجي 🧠"):
                if not gemini_api_key:
                    st.error("أدخل مفتاح Gemini API أولاً.")
                else:
                    with st.spinner("الاستراتيجي يفكر..."):
                        payload = {
                            "match": {"home": home_team_name, "away": away_team_name},
                            "market_analysis": {"fair_probs": fair_h2h, "kelly_suggestions": sugg_h2h},
                            "statistical_analysis": {"poisson_probs": poisson_probs}
                        }
                        try:
                            analysis = analyze_with_gemini(payload=payload)
                            st.markdown(analysis)
                        except Exception as e:
                            st.error(f"حدث خطأ من Gemini: {e}")
