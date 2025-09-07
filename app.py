# app.py (النسخة النهائية مع قراءة Secrets الذكية)
# -*- coding: utf-8 -*-
import os
import streamlit as st
from datetime import datetime

# --- استيراد الدوال من الملفات المساعدة ---
try:
    from odds_math import (
        aggregate_prices, implied_from_decimal, shin_fair_probs, overround,
        kelly_suggestions, normalize_proportional, poisson_prediction
    )
    from gemini_helper import analyze_with_gemini
    import odds_provider_theoddsapi as odds_api
    from stats_fetcher import get_league_stats_from_api
except ImportError as e:
    st.error(f"خطأ في الاستيراد: {e}. تأكد من وجود كل الملفات المساعدة!")
    st.stop()

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Odds Strategist AUTO", page_icon="🧠", layout="wide")

# --- CSS مخصص للتصميم ---
st.markdown("""
<style>
    .prob-bar-container { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
    .prob-bar-title { display: flex; justify-content: space-between; font-size: 0.9em; color: #b0b8c2; }
    .prob-bar { width: 100%; background-color: #334155; border-radius: 5px; overflow: hidden; height: 15px; }
    .prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s ease-in-out; text-align: center; color: white; font-size: 0.8em; font-weight: bold; line-height: 15px; }
</style>
""", unsafe_allow_html=True)

def render_prob_bar(label, probability, color):
    """دالة لرسم شريط الاحتمالات"""
    pct = probability * 100
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-title"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width: {pct}%; background-color: {color};"></div></div>
    </div>
    """

# --- الواجهة الرئيسية ---
st.markdown("<h1>Odds Strategist AUTO 🤖</h1>", unsafe_allow_html=True)
st.markdown("### التحليل الأوتوماتيكي: استراتيجيات السوق + التوقعات الإحصائية (بواسون)")

# --- الدالة الذكية لقراءة المفاتيح ---
def load_api_keys():
    """
    تحميل المفاتيح من Streamlit Secrets أولاً، وإذا لم تجدها، تطلبها من المستخدم.
    """
    st.sidebar.header("🔑 إعدادات المفاتيح")
    
    # Odds API Key
    if 'ODDS_API_KEY' in st.secrets and st.secrets['ODDS_API_KEY']:
        odds_key = st.secrets['ODDS_API_KEY']
        st.sidebar.success("✅ Odds API Key loaded from Secrets.")
    else:
        odds_key = st.sidebar.text_input("The Odds API Key", type="password")
    
    # Gemini API Key
    if 'GEMINI_API_KEY' in st.secrets and st.secrets['GEMINI_API_KEY']:
        gemini_key = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("✅ Gemini API Key loaded from Secrets.")
    else:
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

    # Football Data API Key
    if 'FOOTBALL_DATA_API_KEY' in st.secrets and st.secrets['FOOTBALL_DATA_API_KEY']:
        football_data_key = st.secrets['FOOTBALL_DATA_API_KEY']
        st.sidebar.success("✅ Football Data Key loaded from Secrets.")
    else:
        football_data_key = st.sidebar.text_input("Football Data API Key", type="password")

    # وضع المفاتيح في بيئة التشغيل ليتم استخدامها في الملفات الأخرى
    if odds_key: os.environ["ODDS_API_KEY"] = odds_key
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key

    return odds_key, gemini_key, football_data_key

odds_api_key, gemini_api_key, football_data_key = load_api_keys()

# --- إعدادات المحفظة والسوق ---
st.sidebar.header("🏦 إدارة المحفظة")
bankroll = st.sidebar.number_input("حجم المحفظة ($)", min_value=1.0, value=100.0, step=10.0)
kelly_scale = st.sidebar.slider("معامل كيلي (Kelly Scale)", 0.05, 1.0, 0.25, 0.05)

st.sidebar.header("⚙️ إعدادات السوق")
try:
    if not odds_api_key:
        st.sidebar.warning("الرجاء إدخال مفتاح The Odds API لبدء التحليل.")
        st.stop()
        
    sports = odds_api.list_sports()
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
    if not odds_api_key or not football_data_key:
        st.error("يرجى إدخال كل مفاتيح API المطلوبة في الشريط الجانبي (Odds API و Football Data).")
    else:
        with st.spinner(f"جاري جلب مباريات {selected_sport_label}..."):
            try:
                events, meta = odds_api.fetch_odds_for_sport(sport_key, ",".join(regions), ",".join(markets))
                st.session_state["events_data"] = {"events": events, "meta": meta}
                st.success(f"تم جلب {len(events)} مباراة.")
            except Exception as e:
                st.error(f"حدث خطأ أثناء جلب أسعار المباريات: {e}")
        
        with st.spinner("جاري سحب الإحصائيات من football-data.org..."):
            league_stats = get_league_stats_from_api(api_key=football_data_key, competition_code="PL") 
            if not league_stats:
                st.error("فشل في سحب الإحصائيات.")
            else:
                st.session_state['league_stats'] = league_stats

# --- عرض وتحليل المباريات ---
if "events_data" in st.session_state and "league_stats" in st.session_state:
    events = st.session_state["events_data"]["events"]
    league_stats = st.session_state['league_stats']
    match_options = {f"{ev.get('home_team')} vs {ev.get('away_team')}": i for i, ev in enumerate(events)}
    
    if not match_options:
        st.warning("لم يتم العثور على مباريات.")
    else:
        selected_match_label = st.selectbox("اختر مباراة من القائمة:", list(match_options.keys()))
        event = events[match_options[selected_match_label]]
        
        home_team_name = event['home_team']
        away_team_name = event['away_team']

        tab1, tab2, tab3 = st.tabs(["📊 التحليل المزدوج", "📈 تفاصيل السوق", "🤖 استشارة الخبير Gemini"])

        # 1. تحليل السوق
        h2h_prices = odds_api.extract_h2h_prices(event)
        agg_odds_h2h, imps_h2h, fair_h2h, sugg_h2h = {}, {}, {}, {}
        if any(h2h_prices.values()):
            agg_odds_h2h = {side: aggregate_prices(arr, mode='best') for side, arr in h2h_prices.items()}
            imps_h2h = implied_from_decimal(agg_odds_h2h)
            fair_h2h = shin_fair_probs(imps_h2h)
            sugg_h2h = kelly_suggestions(fair_h2h, agg_odds_h2h, bankroll, kelly_scale)
        
        # 2. التحليل الإحصائي (بواسون)
        home_stats = league_stats.get(home_team_name)
        away_stats = league_stats.get(away_team_name)
        poisson_probs = None
        if home_stats and away_stats:
            poisson_probs = poisson_prediction(
                home_attack=home_stats['attack'], home_defense=home_stats['defense'],
                away_attack=away_stats['attack'], away_defense=away_stats['defense']
            )

        with tab1:
            st.header("مقارنة بين رأي السوق ورأي الإحصاء")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("تحليل السوق (Fair Odds)")
                if fair_h2h:
                    st.markdown(render_prob_bar(home_team_name, fair_h2h.get('home', 0), '#4a90e2'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar("التعادل", fair_h2h.get('draw', 0), '#f5a623'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar(away_team_name, fair_h2h.get('away', 0), '#e24a4a'), unsafe_allow_html=True)
            with col2:
                st.subheader("التحليل الإحصائي (Poisson)")
                if poisson_probs:
                    st.markdown(render_prob_bar(home_team_name, poisson_probs.get('home', 0), '#4a90e2'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar("التعادل", poisson_probs.get('draw', 0), '#f5a623'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar(away_team_name, poisson_probs.get('away', 0), '#e24a4a'), unsafe_allow_html=True)
                else:
                    st.warning(f"لم يتم العثور على إحصائيات لـ '{home_team_name}' أو '{away_team_name}'.")

        with tab2:
            st.header("تحليل سوق نتيجة المباراة (1x2)")
            if not any(s.get('edge', 0) > 0 for s in sugg_h2h.values()):
                st.info("لا توجد فرص قيمة (Value) واضحة في هذا السوق حسب المعايير الحالية.")
            else:
                for side, suggestion in sugg_h2h.items():
                    if suggestion.get('edge', 0) > 0:
                        with st.container(border=True):
                            st.subheader(f"🎯 فرصة قيمة: {side.capitalize()}")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("أفضل سعر في السوق", f"{agg_odds_h2h.get(side, 0):.2f}")
                            c2.metric("الأفضلية (Edge)", f"+{suggestion['edge']*100:.2f}%")
                            c3.metric("الرهان المقترح (كيلي)", f"${suggestion['stake_amount']:.2f}")

        with tab3:
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

