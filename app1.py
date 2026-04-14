# -*- coding: utf-8 -*-
import streamlit as st
import requests
import math
import plotly.graph_objects as go

st.set_page_config(page_title="PRO Mastermind Dashboard", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; }

/* بطاقات الساحر / الخبير (Mastermind View) */
.match-card {
    background: rgba(20, 20, 25, 0.85);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 215, 0, 0.2);
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    border-radius: 20px;
    padding: 35px;
    margin-bottom: 40px;
    direction: rtl;
    color: white;
}

.match-title {
    font-size: 36px;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #FFD700, #FFA500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 30px;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
}

.rec-badge-strong {
    background: linear-gradient(135deg, #11998e, #38ef7d); color: white;
    padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 18px; text-align: right; box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3); border: 1px solid #38ef7d;
}

.rec-badge-golden {
    background: linear-gradient(135deg, #f2c94c, #f2994a); color: #111;
    padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 18px; text-align: right; box-shadow: 0 8px 30px rgba(242, 201, 76, 0.6); border: 2px solid #fff;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(242, 201, 76, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(242, 201, 76, 0); }
    100% { box-shadow: 0 0 0 0 rgba(242, 201, 76, 0); }
}

.rec-badge-extra {
    background: linear-gradient(135deg, #2193b0, #6dd5ed); color: white;
    padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 18px; text-align: right; box-shadow: 0 8px 25px rgba(109, 213, 237, 0.3); border: 1px solid #6dd5ed;
}

.narrative-box {
    background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 15px;
    border-right: 6px solid #FFD700; margin-top: 25px; font-size: 20px; line-height: 1.6;
    box-shadow: inset 0 0 15px rgba(255, 215, 0, 0.05);
}

.rec-header { font-weight: 900; font-size: 22px; margin-bottom: 10px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
.metric-highlight { font-size: 26px; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ آلة الدفع الذكية (Mastermind Edition)")
st.markdown("نسخة **VIP** تعتمد على كسر تشفير الأرقام بمقارنة شركتك المفضلة بالسوق العالمي السري 💸")

API_BASE_URL = "https://api.the-odds-api.com/v4"

def fetch_odds(api_key, sport_key):
    url = f"{API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key, "regions": "eu,uk,us", "markets": "h2h,totals", "oddsFormat": "decimal",
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200: return response.json()
        elif response.status_code == 401: st.error("❌ API Key غير صالح")
        return []
    except Exception as e: return []

def remove_margin(odds_list):
    probs = [1.0/o for o in odds_list if o > 1.0]
    if len(probs) != len(odds_list): return None
    total_margin = sum(probs)
    return [p / total_margin for p in probs]

def get_market_baseline(bookmakers, market_key, point=None):
    """ استخراج المرجع القياسي الصارم (Pinnacle أو متوسط السوق) للحصول على الاحتمال الحقيقي المُخلص من الهامش """
    sharp_bookies = [b for b in bookmakers if b.get('title') == 'Pinnacle']
    target = sharp_bookies[0] if sharp_bookies else None
    
    odds = []
    if target: # التحقق من Pinnacle
        for m in target.get("markets", []):
            if m["key"] == market_key:
                if market_key == "totals" and point:
                    if m.get("outcomes") and m.get("outcomes")[0].get("point") != point: continue
                odds = [out["price"] for out in m["outcomes"]]
                break
    
    # التعويض بمتوسط السوق العالمي في حال عدم وجود Pinnacle
    if not odds or len(odds) < 2:
        avgs = {}
        counts = {}
        for b in bookmakers:
            for m in b.get("markets", []):
                if m["key"] == market_key:
                    if market_key == "totals" and point:
                        if m.get("outcomes") and m.get("outcomes")[0].get("point") != point: continue
                    for i, out in enumerate(m["outcomes"]):
                        avgs[i] = avgs.get(i, 0) + out["price"]
                        counts[i] = counts.get(i, 0) + 1
        if counts:
            odds = [avgs[i]/counts[i] for i in range(len(avgs))]
            
    if len(odds) >= 2:
        return remove_margin(odds)
    return None

def extract_target_odds(bookmakers, market_key, target_bookie, point=None):
    """ استخراج أودز الشركة المستهدفة فقط للرهان """
    for b in bookmakers:
        if b.get("title") == target_bookie:
            for m in b.get("markets", []):
                if m["key"] == market_key:
                    if market_key == "totals" and point:
                        if m.get("outcomes") and m.get("outcomes")[0].get("point") != point: continue
                    return {out["name"]: out["price"] for out in m["outcomes"]}
    return {}

def calculate_dynamic_stake(true_prob, odds, bankroll, fraction):
    """ حساب حجم الرهان المثالي بكسر ميزانية ذكي """
    b = odds - 1.0
    q = 1.0 - true_prob
    if b <= 0: return 0
    kelly = (b * true_prob - q) / b
    if kelly <= 0: return 0
    stake = bankroll * kelly * fraction
    
    # حساب قيمة الرهان مع وضع حدود دنيا وعليا للسلامة
    calculated_stake = min(max(stake, bankroll * 0.02), bankroll * 0.15)
    
    # فرض الحد الأدنى المسموح للرهان بقيمة 22
    return max(calculated_stake, 22.0)

def render_gauge(prob, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': '%', 'font': {'size': 35, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FFD700" if prob > 0.45 else "grey"},
            'bgcolor': "rgba(0,0,0,0.3)", 'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "rgba(255, 0, 0, 0.2)"},
                {'range': [40, 60], 'color': "rgba(255, 165, 0, 0.2)"},
                {'range': [60, 100], 'color': "rgba(0, 255, 0, 0.2)"}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

with st.sidebar:
    st.header("⚙️ مركز القيادة المركزي")
    api_key_input = st.text_input("Odds API Key", type="password")
    
    target_bookmaker = st.radio("🎯 المنصة المعتمدة للمراهنة", ["1xBet", "Melbet"])
    
    sports = {
        "الدوري الإنجليزي": "soccer_epl", "الدوري الإسباني": "soccer_spain_la_liga",
        "الدوري الألماني": "soccer_germany_bundesliga", "الدوري الإيطالي": "soccer_italy_serie_a",
        "دوري أبطال أوروبا": "soccer_uefa_champs_league", "كرة السلة - NBA": "basketball_nba"
    }
    selected_sport = st.selectbox("اختر البطولة", list(sports.keys()))
    sport_key = sports[selected_sport]
    
    st.markdown("---")
    st.header("💰 الخزينة (Bankroll)")
    user_bankroll = st.number_input("مقدار الرصيد الحالي", min_value=100.0, max_value=100000.0, value=500.0, step=100.0)
    risk_level = st.selectbox("أسلوب الرادار الاستثماري", ["استثمار آمن", "متوازن", "هجوم مغامر"], index=1)
    
    kelly_fraction_map = {"استثمار آمن": 0.20, "متوازن": 0.35, "هجوم مغامر": 0.50}
    active_fraction = kelly_fraction_map[risk_level]

if st.button("🚀 تشغيل رادار الآلة الحاسبة الخفية", type="primary", use_container_width=True):
    if not api_key_input: st.warning("الرجاء إدخال API Key لبدء الاتصال بالخوادم.")
    else:
        with st.spinner("جارِ دمج البيانات، كسر شفرات التسعير، واستخراج الفرص الذهبية..."):
            matches = fetch_odds(api_key_input, sport_key)
            analyzed_count = 0
            
            for index, match in enumerate(matches):
                home_team = match["home_team"]
                away_team = match["away_team"]
                bookmakers = match.get("bookmakers", [])
                
                # المعايرة بالسوق الحقيقي
                baseline_1x2 = get_market_baseline(bookmakers, "h2h")
                baseline_ou = get_market_baseline(bookmakers, "totals", point=2.5)
                
                # استخراج أسعار الشركة المحددة من المستخدم
                target_1x2 = extract_target_odds(bookmakers, "h2h", target_bookmaker)
                target_ou = extract_target_odds(bookmakers, "totals", target_bookmaker, point=2.5)
                
                recs = []
                home_prob_disp, away_prob_disp = 0, 0
                
                if baseline_1x2 and target_1x2 and home_team in target_1x2 and away_team in target_1x2:
                    h_true, d_true, a_true = (baseline_1x2[0], baseline_1x2[1], baseline_1x2[2]) if len(baseline_1x2)==3 else (baseline_1x2[0], 0, baseline_1x2[1])
                    home_prob_disp, away_prob_disp = h_true, a_true
                    
                    ho = target_1x2[home_team]
                    ao = target_1x2[away_team]
                    
                    outcomes = [("Home", home_team, ho, h_true), ("Away", away_team, ao, a_true)]
                    is_balanced = abs(ho - ao) <= 0.2
                    
                    for key, team, odds, t_prob in outcomes:
                        ev = (t_prob * odds) - 1.0
                        
                        # 1. صيد الفرصة الذهبية 💎 (أداة صنع النقود) - إذا كانت الشركة تدفع أكثر من الاحتمال الحقيقي للسوق المفتوح
                        if ev > 0.02 and 1.30 <= odds <= 3.00:
                            stake = calculate_dynamic_stake(t_prob, odds, user_bankroll, active_fraction)
                            if stake > 0:
                                recs.append({"label": "💎 فرصة ذهبية مخفية (خطأ تسعير!):", "bet": f"فوز {team}", "odds": odds, "stake": round(stake*1.2), "priority": 0, "ev": ev})
                        
                        # 2. توصية قوية تقليدية
                        elif not is_balanced and odds < 1.70:
                            stake = calculate_dynamic_stake(t_prob + 0.05, odds, user_bankroll, active_fraction)
                            recs.append({"label": "🟢 قوة ضاربة:", "bet": f"فوز {team}", "odds": odds, "stake": round(stake), "priority": 1, "ev": 0})
                            
                        # 3. توصية قيمة عادية
                        elif not is_balanced and 2.00 < odds <= 3.00 and ev > -0.05:
                            stake = calculate_dynamic_stake(t_prob + 0.03, odds, user_bankroll, active_fraction)
                            recs.append({"label": "🟡 توصية قيمة (مخاطرة متوازنة):", "bet": f"فوز {team}", "odds": odds, "stake": round(stake), "priority": 2, "ev": 0})
                
                # الأهداف
                ou_narrative = ""
                if baseline_ou and target_ou and "Over" in target_ou and "Under" in target_ou:
                    ov_true, un_true = baseline_ou
                    oo = target_ou["Over"]
                    uo = target_ou["Under"]
                    
                    for title, odds, t_prob in [("أكثر من 2.5 هدف", oo, ov_true), ("أقل من 2.5 هدف", uo, un_true)]:
                        ev = (t_prob * odds) - 1.0
                        if ev > 0.02 and 1.30 <= odds <= 3.00:
                            stake = calculate_dynamic_stake(t_prob, odds, user_bankroll, active_fraction)
                            recs.append({"label": "💎 صيدة أهداف لا تُفوت (Value!):", "bet": title, "odds": odds, "stake": round(stake*1.2), "priority": 0, "ev": ev})
                        elif 1.30 <= odds <= 1.80 and ev > -0.05:
                            stake = calculate_dynamic_stake(t_prob + 0.04, odds, user_bankroll, active_fraction)
                            pr = 2 if not any(r["priority"]==2 for r in recs) else 3
                            lbl = "🟡 اقتناص الأهداف:" if pr==2 else "🔵 خيار مساند:"
                            recs.append({"label": lbl, "bet": title, "odds": odds, "stake": round(stake), "priority": pr, "ev": 0})
                            
                    if oo < uo: ou_narrative = "مفتوح"
                    else: ou_narrative = "مغلق"
                
                final_recs = sorted(recs, key=lambda x: (-x["ev"], x["priority"]))[:3]
                
                # توليد قراءة العقل المدبر (النص السري) لتوقع السيناريو
                narrative_text = ""
                if final_recs and home_prob_disp > 0 and away_prob_disp > 0:
                    team_fav = home_team if home_prob_disp > away_prob_disp else away_team
                    prob_fav = max(home_prob_disp, away_prob_disp)
                    
                    if prob_fav > 0.60 and ou_narrative == "مغلق":
                        narrative_text = f"💡 **قراءة ما بين السطور:** الأرقام تشير لاحتكار مستحوذ من {team_fav} مع عقم هجومي أو دفاع صارم للخصم. بوكي {target_bookmaker} يتوقع انتصاراً متواضعاً (1-0 أو 2-0). الرهان على فوزهم مع 'أقل من 3.5 أهداف' يُعد فخاً رياضياً ذهبياً!"
                    elif prob_fav > 0.60 and ou_narrative == "مفتوح":
                        narrative_text = f"💡 **قراءة ما بين السطور:** إعصار هجومي متوقع لاكتساح الملعب. {team_fav} يستند على تفوق هجومي جلي لتدمير الخصم مبكراً. فرصة استثنائية لدمج انتصارهم مع (أكثر من 2.5 أهداف)."
                    elif abs(home_prob_disp - away_prob_disp) < 0.10:
                        narrative_text = f"💡 **قراءة ما بين السطور:** معركة حربية طاحنة تتساوى فيها الحظوظ. السوق العقلاني عاجز عن التوقع الدقيق. أذكى حركة هنا هي الابتعاد الكلي عن تحديد الفائز والتركيز الخالص على الأهداف."

                if final_recs:
                    analyzed_count += 1
                    with st.container():
                        st.markdown(f'<div class="match-card"><div class="match-title">{home_team} ⚔️ {away_team}</div>', unsafe_allow_html=True)
                        
                        g_col1, g_col2 = st.columns(2)
                        with g_col1:
                            f1 = render_gauge(home_prob_disp, f"السوق القياسي: فوز {home_team}")
                            st.plotly_chart(f1, use_container_width=True, key=f"gu1_{index}_{home_team}")
                        with g_col2:
                            f2 = render_gauge(away_prob_disp, f"السوق القياسي: فوز {away_team}")
                            st.plotly_chart(f2, use_container_width=True, key=f"gu2_{index}_{away_team}")
                        
                        if narrative_text:
                            st.markdown(f'<div class="narrative-box">{narrative_text}</div>', unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        r_cols = st.columns(len(final_recs))
                        for i, r in enumerate(final_recs):
                            with r_cols[i]:
                                if "💎" in r['label']: b_class = "rec-badge-golden"
                                elif "🟢" in r['label']: b_class = "rec-badge-strong"
                                elif "🟡" in r['label']: b_class = "rec-badge-value"
                                else: b_class = "rec-badge-extra"
                                
                                st.markdown(f"""
                                <div class="{b_class}">
                                    <div class="rec-header">{r['label']}</div>
                                    راهن بمقدار <span class="metric-highlight">{r['stake']}</span> وحدة على:<br>
                                    <div style="font-size:24px; font-weight:900; margin: 15px 0;">{r['bet']}</div>
                                    أودز {target_bookmaker}: <span class="metric-highlight">{r['odds']:.2f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            if analyzed_count == 0 and matches:
                st.info("⚠️ الرادار المستقل لم يلتقط، للأسف، أي فرصة تكسر قواعد التسعير لشركات المراهنة في الوقت الحالي. الخسارة المحتملة أعلى من الربح هنا.")
