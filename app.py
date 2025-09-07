# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
from datetime import datetime
import pandas as pd

# --- استيراد الدوال ---
try:
    from odds_math import *
    from gemini_helper import analyze_with_gemini
    import odds_provider_theoddsapi as odds_api
    from data_loader import load_stats_data_from_csv
except ImportError as e:
    st.error(f"خطأ في الاستيراد: {e}. تأكد من وجود كل الملفات المساعدة!")
    st.stop()

# --- إعدادات الصفحة والـ CSS ---
st.set_page_config(page_title="Odds Strategist - Council of Experts", page_icon="🏛️", layout="wide")

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
st.markdown("<h1>Odds Strategist - مجلس الخبراء 🏛️</h1>", unsafe_allow_html=True)

# --- الشريط الجانبي ---
def load_api_keys():
    st.sidebar.header("🔑 إعدادات المفاتيح")
    odds_key, gemini_key = None, None
    if 'ODDS_API_KEY' in st.secrets:
        odds_key = st.secrets['ODDS_API_KEY']
        st.sidebar.success("✅ Odds API Key loaded.")
    else:
        odds_key = st.sidebar.text_input("The Odds API Key", type="password")
    if 'GEMINI_API_KEY' in st.secrets:
        gemini_key = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("✅ Gemini API Key loaded.")
    else:
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    if odds_key: os.environ["ODDS_API_KEY"] = odds_key
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key
    return odds_key, gemini_key

odds_api_key, gemini_api_key = load_api_keys()

st.sidebar.header("🏦 إدارة المحفظة")
bankroll = st.sidebar.number_input("حجم المحفظة ($)", 1.0, value=100.0, step=10.0)
kelly_scale = st.sidebar.slider("معامل كيلي (Kelly Scale)", 0.05, 1.0, 0.25, 0.05)

st.sidebar.header("⚙️ إعدادات السوق")
try:
    if not odds_api_key:
        st.sidebar.warning("الرجاء إدخال مفتاح The Odds API.")
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

st.sidebar.header("📊 مصدر بيانات الإحصائيات")
stats_csv_path = st.sidebar.text_input("مسار ملف CSV للإحصائيات", "E0.csv", help="حمّل الملف من football-data.co.uk وضعه في نفس المجلد.")

# --- جلب البيانات ---
if st.button("🚀 جلب وتحليل المباريات"):
    if not odds_api_key:
        st.error("يرجى إدخال مفتاح The Odds API.")
    else:
        with st.spinner(f"جاري جلب مباريات {selected_sport_label}..."):
            try:
                events, meta = odds_api.fetch_odds_for_sport(sport_key, ",".join(regions), ",".join(markets))
                st.session_state["events_data"] = {"events": events, "meta": meta}
                st.success(f"تم جلب {len(events)} مباراة.")
            except Exception as e:
                st.error(f"حدث خطأ أثناء جلب أسعار المباريات: {e}")
                st.session_state["events_data"] = None
        
        with st.spinner("جاري تحميل ومعالجة ملف الإحصائيات..."):
            if os.path.exists(stats_csv_path):
                stats_df = load_stats_data_from_csv(stats_csv_path)
                st.session_state['stats_df'] = stats_df
                if stats_df is not None:
                    st.success("تم تحميل ملف الإحصائيات بنجاح.")
                else:
                    st.error("فشل في قراءة ملف الإحصائيات.")
            else:
                st.error(f"ملف الإحصائيات '{stats_csv_path}' غير موجود.")
                st.session_state['stats_df'] = None

# --- عرض وتحليل المباريات ---
if "events_data" in st.session_state and st.session_state["events_data"]:
    events = st.session_state["events_data"]["events"]
    stats_df = st.session_state.get('stats_df')
    match_options = {f"{ev.get('home_team')} vs {ev.get('away_team')}": i for i, ev in enumerate(events)}
    
    if match_options:
        selected_match_label = st.selectbox("اختر مباراة من القائمة:", list(match_options.keys()))
        event = events[match_options[selected_match_label]]
        
        home_team_name = event['home_team']
        away_team_name = event['away_team']

        # --- تشغيل مجلس الخبراء ---
        # 1. تحليل السوق
        h2h_prices = odds_api.extract_h2h_prices(event)
        agg_odds_h2h, fair_h2h, sugg_h2h = {}, {}, {}
        if any(h2h_prices.values()):
            agg_odds_h2h = {s: aggregate_prices(p, 'best') for s, p in h2h_prices.items()}
            fair_h2h = shin_fair_probs(implied_from_decimal(agg_odds_h2h))
            sugg_h2h = kelly_suggestions(fair_h2h, agg_odds_h2h, bankroll, kelly_scale)
        
        # 2, 3, 4. التحليلات الإحصائية
        poisson_probs, form_probs, xg_probs = None, None, None
        if stats_df is not None:
            poisson_probs = poisson_prediction(home_team_name, away_team_name, stats_df)
            form_probs = calculate_form_probs(home_team_name, away_team_name, stats_df)
            xg_probs = calculate_xg_probs(home_team_name, away_team_name, stats_df)

        # --- عرض الواجهة بالتابات ---
        tab1, tab2, tab3, tab4 = st.tabs(["🏛️ مجلس الخبراء", "📈 تفاصيل 1x2", "⚽️ تفاصيل الأهداف", "🤖 استشارة Gemini"])

        with tab1:
            st.header("آراء مجلس الخبراء")
            def get_verdict(probs):
                if not probs: return "بيانات ناقصة"
                max_prob = max(probs, key=probs.get)
                if max_prob == 'home': return f"فوز {home_team_name}"
                if max_prob == 'away': return f"فوز {away_team_name}"
                return "التعادل"

            verdicts = {
                "market": get_verdict(fair_h2h),
                "poisson": get_verdict(poisson_probs),
                "form": get_verdict(form_probs),
                "xg": get_verdict(xg_probs)
            }

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.subheader("👨‍💼 خبير السوق")
                st.metric("يرشح:", verdicts["market"])
            with col2:
                st.subheader("🎯 خبير الأهداف")
                st.metric("يرشح:", verdicts["poisson"])
            with col3:
                st.subheader("📈 خبير الأداء الحالي")
                st.metric("يرشح:", verdicts["form"])
            with col4:
                st.subheader("🔬 خبير الأداء النوعي")
                st.metric("يرشح:", verdicts["xg"])
            
            st.markdown("---")
            st.header("⭐ الخلاصة النهائية ومؤشر الثقة")

            votes = [v for v in verdicts.values() if v != "بيانات ناقصة"]
            if len(votes) > 0:
                most_common_verdict = max(set(votes), key=votes.count)
                num_votes = votes.count(most_common_verdict)
                st.metric(f"النتيجة الأكثر ترجيحًا:", f"{most_common_verdict}", f"{num_votes} / {len(votes)} خبراء يتفقون")
            else:
                st.warning("لا يمكن حساب مؤشر الثقة بسبب نقص كل التحليلات.")

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
            st.header("تحليل سوق الأهداف (Over/Under)")
            totals_lines = odds_api.extract_totals_lines(event)
            if not totals_lines:
                st.info("لا توجد بيانات لسوق الأهداف لهذه المباراة.")
            else:
                selected_line = st.selectbox("اختر خط الأهداف:", sorted(totals_lines.keys(), key=float))
                line_data = totals_lines[selected_line]
                agg_odds_ou = {'over': aggregate_prices(line_data.get('over', []), 'best'), 'under': aggregate_prices(line_data.get('under', []), 'best')}
                if agg_odds_ou['over'] > 0 and agg_odds_ou['under'] > 0:
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
                else:
                    st.warning("لا توجد أسعار كافية لتحليل هذا الخط.")

        with tab4:
            if st.button("اطلب تحليلاً مفصلاً من رئيس المجلس 🧠"):
                if not gemini_api_key:
                    st.error("أدخل مفتاح Gemini API أولاً.")
                else:
                    with st.spinner("الخبير الاستراتيجي يفكر..."):
                        payload = {
                            "match": {"home": home_team_name, "away": away_team_name},
                            "market_analysis": {"verdict": verdicts["market"], "fair_probs": fair_h2h},
                            "poisson_analysis": {"verdict": verdicts["poisson"], "probs": poisson_probs},
                            "form_analysis": {"verdict": verdicts["form"], "probs": form_probs},
                            "xg_analysis": {"verdict": verdicts["xg"], "probs": xg_probs}
                        }
                        try:
                            analysis = analyze_with_gemini(payload=payload)
                            st.markdown(analysis)
                        except Exception as e:
                            st.error(f"حدث خطأ من Gemini: {e}")
