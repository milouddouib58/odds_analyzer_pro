# app.py (النسخة المصححة)
# -*- coding: utf-8 -*-
import os
import streamlit as st
from datetime import datetime

# --- استيراد الدوال من الملفات المساعدة ---
try:
    from odds_math import (
        aggregate_prices, implied_from_decimal, shin_fair_probs, overround,
        kelly_suggestions, normalize_proportional, analyze_market_depth
    )
    from gemini_helper import analyze_with_gemini
    import odds_provider_theoddsapi as odds_api
except ImportError as e:
    st.error(f"خطأ في الاستيراد: {e}. تأكد من وجود كل الملفات المساعدة!")
    st.stop()

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Odds Analyzer PRO", page_icon="🏆", layout="wide")

# --- CSS مخصص للتصميم الاحترافي ---
st.markdown("""
<style>
    .prob-bar-container { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
    .prob-bar-title { display: flex; justify-content: space-between; font-size: 0.9em; color: #b0b8c2; }
    .prob-bar { width: 100%; background-color: #334155; border-radius: 5px; overflow: hidden; height: 15px; }
    .prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s ease-in-out; text-align: center; color: white; font-size: 0.8em; font-weight: bold; line-height: 15px; }
</style>
""", unsafe_allow_html=True)

def render_prob_bar(label, probability, color):
    pct = probability * 100
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-title"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width: {pct}%; background-color: {color};"></div></div>
    </div>
    """

# --- الواجهة الرئيسية ---
st.markdown("<h1>Odds Analyzer PRO 🏆</h1>", unsafe_allow_html=True)
st.markdown("### التحليل الاحترافي للأسواق: القيمة (Edge), إدارة المخاطر (Kelly), والذكاء الاصطناعي")

# --- الشريط الجانبي للإعدادات ---
st.sidebar.header("🔑 إعدادات المفاتيح")
odds_api_key = st.sidebar.text_input("The Odds API Key", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if odds_api_key: os.environ["ODDS_API_KEY"] = odds_api_key
if gemini_api_key: os.environ["GEMINI_API_KEY"] = gemini_api_key

st.sidebar.header("🏦 إدارة المحفظة")
bankroll = st.sidebar.number_input("حجم المحفظة ($)", min_value=1.0, value=100.0, step=10.0)
kelly_scale = st.sidebar.slider("معامل كيلي (Kelly Scale)", 0.05, 1.0, 0.25, 0.05, help="لتخفيف المخاطرة, استخدم قيمة أقل من 1 (مثلاً 0.25 لربع كيلي).")

st.sidebar.header("⚙️ إعدادات السوق")
try:
    sports = odds_api.list_sports()
    sport_options = {f"{s.get('group')} - {s.get('title')}": s.get("key") for s in sports}
    selected_sport_label = st.sidebar.selectbox("اختر الرياضة:", list(sport_options.keys()))
    sport_key = sport_options[selected_sport_label]
except Exception:
    st.sidebar.error("لا يمكن جلب الرياضات. تأكد من مفتاح API.")
    st.stop()

regions = st.sidebar.multiselect("المناطق:", ["eu", "uk", "us", "au"], default=["eu", "uk"])
markets = st.sidebar.multiselect("الأسواق:", ["h2h", "totals"], default=["h2h", "totals"])

# --- جلب البيانات ---
if st.button("🚀 جلب وتحليل المباريات"):
    if not os.getenv("ODDS_API_KEY"):
        st.error("يرجى إدخال مفتاح The Odds API في الشريط الجانبي.")
    else:
        with st.spinner(f"جاري جلب مباريات {selected_sport_label}..."):
            try:
                events, meta = odds_api.fetch_odds_for_sport(sport_key, ",".join(regions), ",".join(markets))
                st.session_state["events_data"] = {"events": events, "meta": meta}
                st.success(f"تم جلب {len(events)} مباراة. (الطلبات المتبقية: {meta.get('requests_remaining')})")
            except Exception as e:
                st.error(f"حدث خطأ: {e}")

# --- عرض وتحليل المباريات ---
if "events_data" in st.session_state:
    events = st.session_state["events_data"]["events"]
    match_options = {f"{ev.get('home_team')} vs {ev.get('away_team')}": i for i, ev in enumerate(events)}
    
    if not match_options:
        st.warning("لم يتم العثور على مباريات.")
    else:
        selected_match_label = st.selectbox("اختر مباراة من القائمة:", list(match_options.keys()))
        event = events[match_options[selected_match_label]]
        tab1, tab2, tab3, tab4 = st.tabs(["📊 نظرة عامة", "📈 تحليل 1x2", "⚽️ تحليل الأهداف", "🤖 تحليل Gemini"])

        h2h_prices = odds_api.extract_h2h_prices(event)
        if any(h2h_prices.values()):
            agg_odds_h2h = {side: aggregate_prices(arr, mode='best') for side, arr in h2h_prices.items()}
            imps_h2h = implied_from_decimal(agg_odds_h2h)
            fair_h2h = shin_fair_probs(imps_h2h)
            sugg_h2h = kelly_suggestions(fair_h2h, agg_odds_h2h, bankroll, kelly_scale)
        else:
            agg_odds_h2h, imps_h2h, fair_h2h, sugg_h2h = {}, {}, {}, {}

        totals_lines = odds_api.extract_totals_lines(event)

        with tab1:
            st.header(f"ملخص مباراة: {selected_match_label}")
            if fair_h2h:
                st.subheader("الاحتمالات العادلة (Fair Probs)")
                col1, _ = st.columns([2,1])
                with col1:
                    st.markdown(render_prob_bar(event['home_team'], fair_h2h.get('home', 0), '#4a90e2'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar("التعادل", fair_h2h.get('draw', 0), '#f5a623'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar(event['away_team'], fair_h2h.get('away', 0), '#e24a4a'), unsafe_allow_html=True)
                st.info(f"هامش الربح (Overround) لسوق 1x2 هو: **{overround(imps_h2h):.3f}**")
        with tab2:
            st.header("تحليل سوق نتيجة المباراة (1x2)")
            if not any(s.get('edge', 0) > 0 for s in sugg_h2h.values()):
                st.info("لا توجد فرص قيمة (Value) واضحة في هذا السوق حسب المعايير الحالية.")
            else:
                for side, suggestion in sugg_h2h.items():
                    if suggestion.get('edge', 0) > 0:
                        with st.container(border=True):
                            st.subheader(f"🎯 فرصة قيمة: {side.capitalize()}")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("أفضل سعر في السوق", f"{agg_odds_h2h.get(side, 0):.2f}")
                            col2.metric("الأفضلية (Edge)", f"+{suggestion['edge']*100:.2f}%")
                            col3.metric("الرهان المقترح (كيلي)", f"${suggestion['stake_amount']:.2f}")
            
            st.markdown("---")
            st.header("🔬 تحليل عمق السوق (Market Depth)")
            with st.expander("شرح تحليل العمق"):
                # --- ::: This is the corrected section ::: ---
                st.write("""
                هذا التحليل يقيس "إجماع" السوق.
                - **متوسط السعر:** يعطيك السعر الوسطي لكل شركات المراهنات.
                - **تشتت الأسعار (الانحراف):** رقم صغير يعني أن الشركات متفقة, ورقم كبير يعني أن هناك اختلافًا في الآراء وفرصًا محتملة.
                - **الأسعار الشاردة:** هي أسعار عالية جدًا مقارنة ببقية السوق, وتمثل أفضل الفرص.
                """)
            
            if h2h_prices:
                col1, col2, col3 = st.columns(3)
                sides = {'home': event['home_team'], 'draw': 'التعادل', 'away': event['away_team']}
                for key, name in sides.items():
                    if h2h_prices.get(key):
                        analysis = analyze_market_depth(h2h_prices[key])
                        container = col1 if key == 'home' else col2 if key == 'draw' else col3
                        with container:
                            st.subheader(name)
                            st.metric(label="متوسط السعر", value=f"{analysis['mean']:.2f}")
                            st.metric(label="تشتت الأسعار", value=f"{analysis['std_dev']:.2f}")
                            if analysis['outliers']:
                                st.success(f"🚨 أسعار شاردة: {analysis['outliers']}")

        with tab3:
            st.header("تحليل سوق الأهداف (Over/Under)")
            if not totals_lines:
                st.info("لا توجد بيانات لسوق الأهداف لهذه المباراة.")
            else:
                selected_line = st.selectbox("اختر خط الأهداف:", sorted(totals_lines.keys(), key=float))
                line_data = totals_lines[selected_line]
                agg_odds_ou = {'over': aggregate_prices(line_data['over'], 'best'), 'under': aggregate_prices(line_data['under'], 'best')}
                imps_ou = implied_from_decimal(agg_odds_ou)
                fair_ou = shin_fair_probs(imps_ou)
                sugg_ou = kelly_suggestions(fair_ou, agg_odds_ou, bankroll, kelly_scale)
                st.subheader(f"الاحتمالات العادلة لخط {selected_line}")
                st.markdown(render_prob_bar(f"Over {selected_line}", fair_ou.get('over', 0), '#22c55e'), unsafe_allow_html=True)
                st.markdown(render_prob_bar(f"Under {selected_line}", fair_ou.get('under', 0), '#ef4444'), unsafe_allow_html=True)
                if any(s.get('edge', 0) > 0 for s in sugg_ou.values()):
                    for side, suggestion in sugg_ou.items():
                         if suggestion.get('edge', 0) > 0:
                             with st.container(border=True):
                                st.subheader(f"🎯 فرصة قيمة: {side.capitalize()} {selected_line}")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("أفضل سعر", f"{agg_odds_ou.get(side, 0):.2f}")
                                c2.metric("الأفضلية (Edge)", f"+{suggestion['edge']*100:.2f}%")
                                c3.metric("الرهان المقترح", f"${suggestion['stake_amount']:.2f}")
        with tab4:
            st.header("🤖 تحليل معمق بواسطة Gemini")
            if st.button("اطلب من Gemini تحليل المباراة"):
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("يرجى إدخال مفتاح Gemini API.")
                else:
                    with st.spinner("Gemini يحلل البيانات..."):
                        payload = {"match": {"home": event['home_team'], "away": event['away_team']}, "h2h": {"odds": agg_odds_h2h, "fair_probs": fair_h2h, "kelly": sugg_h2h}}
                        try:
                            analysis = analyze_with_gemini(payload=payload)
                            st.markdown(analysis)
                        except Exception as e:
                            st.error(f"حدث خطأ من Gemini: {e}")
