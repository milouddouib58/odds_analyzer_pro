# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
from datetime import datetime
import pandas as pd

# --- Importar funciones ---
try:
    from odds_math import *
    from gemini_helper import analyze_with_gemini
    import odds_provider_theoddsapi as odds_api
    from data_loader import load_stats_data_from_csv
except ImportError as e:
    st.error(f"Error en la importación: {e}. ¡Asegúrese de que todos los archivos auxiliares existan!")
    st.stop()

# --- Configuración de la página y CSS ---
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
    """Función para dibujar la barra de probabilidad"""
    pct = probability * 100
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-title"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width: {pct}%; background-color: {color};"></div></div>
    </div>
    """

# --- Interfaz principal ---
st.markdown("<h1>Odds Strategist - Consejo de Expertos 🏛️</h1>", unsafe_allow_html=True)

# --- Barra lateral ---
def load_api_keys():
    st.sidebar.header("🔑 Configuración de claves API")
    odds_key, gemini_key = None, None
    if 'ODDS_API_KEY' in st.secrets:
        odds_key = st.secrets['ODDS_API_KEY']
        st.sidebar.success("✅ Clave API de Odds cargada.")
    else:
        odds_key = st.sidebar.text_input("Clave API de The Odds", type="password")
    if 'GEMINI_API_KEY' in st.secrets:
        gemini_key = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("✅ Clave API de Gemini cargada.")
    else:
        gemini_key = st.sidebar.text_input("Clave API de Gemini", type="password")
    if odds_key: os.environ["ODDS_API_KEY"] = odds_key
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key
    return odds_key, gemini_key

odds_api_key, gemini_api_key = load_api_keys()

st.sidebar.header("🏦 Gestión de fondos")
bankroll = st.sidebar.number_input("Tamaño del fondo ($)", 1.0, value=100.0, step=10.0)
kelly_scale = st.sidebar.slider("Factor Kelly (Kelly Scale)", 0.05, 1.0, 0.25, 0.05)

st.sidebar.header("⚙️ Configuración del mercado")
try:
    if not odds_api_key:
        st.sidebar.warning("Por favor, ingrese la clave API de The Odds.")
        st.stop()
    sports = odds_api.list_sports()
    sport_options = {f"{s.get('group')} - {s.get('title')}": s.get("key") for s in sports}
    selected_sport_label = st.sidebar.selectbox("Seleccione el deporte:", list(sport_options.keys()))
    sport_key = sport_options[selected_sport_label]
    regions = st.sidebar.multiselect("Regiones:", ["eu", "uk", "us", "au"], default=["eu", "uk"])
    markets = st.sidebar.multiselect("Mercados:", ["h2h", "totals"], default=["h2h", "totals"])
except Exception as e:
    st.error(f"No se pueden obtener los deportes. Verifique la clave API de The Odds. Error: {e}")
    st.stop()

st.sidebar.header("📊 Fuente de datos de estadísticas")
st.sidebar.info("Seleccione el archivo CSV que ha descargado de football-data.co.uk")
available_csv_files = {
    "Premier League (E0.csv)": "E0.csv",
    "La Liga (SP1.csv)": "SP1.csv",
    "Serie A (I1.csv)": "I1.csv",
    "Bundesliga (D1.csv)": "D1.csv",
    "Ligue 1 (F1.csv)": "F1.csv",
}
selected_csv_label = st.sidebar.selectbox("Seleccione el archivo de la liga:", list(available_csv_files.keys()))
stats_csv_path = available_csv_files[selected_csv_label]

# --- Obtener datos ---
if st.button("🚀 Obtener y analizar partidos"):
    if not odds_api_key:
        st.error("Por favor, ingrese la clave API de The Odds.")
    else:
        with st.spinner(f"Obteniendo partidos para {selected_sport_label}..."):
            try:
                events, meta = odds_api.fetch_odds_for_sport(sport_key, ",".join(regions), ",".join(markets))
                st.session_state["events_data"] = {"events": events, "meta": meta}
                st.success(f"Se obtuvieron {len(events)} partidos.")
            except Exception as e:
                st.error(f"Error al obtener las cuotas de los partidos: {e}")
                st.session_state["events_data"] = None
        
        with st.spinner("Cargando y procesando el archivo de estadísticas..."):
            if os.path.exists(stats_csv_path):
                stats_df = load_stats_data_from_csv(stats_csv_path)
                st.session_state['stats_df'] = stats_df
                if stats_df is not None:
                    st.success(f"Archivo '{stats_csv_path}' cargado exitosamente.")
                else:
                    st.error(f"Error al leer el archivo '{stats_csv_path}'.")
            else:
                st.error(f"El archivo '{stats_csv_path}' no se encuentra. Por favor, descárguelo y colóquelo en la carpeta.")
                st.session_state['stats_df'] = None

# --- Mostrar y analizar partidos ---
if "events_data" in st.session_state and st.session_state["events_data"]:
    events = st.session_state["events_data"]["events"]
    stats_df = st.session_state.get('stats_df')
    match_options = {f"{ev.get('home_team')} vs {ev.get('away_team')}": i for i, ev in enumerate(events)}
    
    if match_options:
        selected_match_label = st.selectbox("Seleccione un partido de la lista:", list(match_options.keys()))
        event = events[match_options[selected_match_label]]
        
        home_team_name = event['home_team']
        away_team_name = event['away_team']

        # --- Ejecutar el consejo de expertos ---
        # 1. Análisis de mercado
        h2h_prices = odds_api.extract_h2h_prices(event)
        agg_odds_h2h, fair_h2h, sugg_h2h = {}, {}, {}
        if any(h2h_prices.values()):
            agg_odds_h2h = {s: aggregate_prices(p, 'best') for s, p in h2h_prices.items()}
            fair_h2h = shin_fair_probs(implied_from_decimal(agg_odds_h2h))
            sugg_h2h = kelly_suggestions(fair_h2h, agg_odds_h2h, bankroll, kelly_scale)
        
        # 2, 3, 4. Análisis estadísticos
        poisson_probs, form_probs, xg_probs = None, None, None
        if stats_df is not None:
            poisson_probs = poisson_prediction(home_team_name, away_team_name, stats_df)
            form_probs = calculate_form_probs(home_team_name, away_team_name, stats_df)
            xg_probs = calculate_xg_probs(home_team_name, away_team_name, stats_df)

        # --- Mostrar la interfaz con pestañas ---
        tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Consejo de Expertos", "📈 Detalles 1x2", "⚽️ Detalles de Goles", "🤖 Consulta a Gemini"])

        with tab1:
            st.header("Opiniones del Consejo de Expertos")
            def get_verdict(probs):
                if not probs: return "Datos insuficientes"
                max_prob = max(probs, key=probs.get)
                if max_prob == 'home': return f"Victoria de {home_team_name}"
                if max_prob == 'away': return f"Victoria de {away_team_name}"
                return "Empate"

            verdicts = {
                "market": get_verdict(fair_h2h),
                "poisson": get_verdict(poisson_probs),
                "form": get_verdict(form_probs),
                "xg": get_verdict(xg_probs)
            }

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.subheader("👨‍💼 Experto de Mercado")
                st.metric("Recomienda:", verdicts["market"])
            with col2:
                st.subheader("🎯 Experto de Goles")
                st.metric("Recomienda:", verdicts["poisson"])
            with col3:
                st.subheader("📈 Experto de Forma Actual")
                st.metric("Recomienda:", verdicts["form"])
            with col4:
                st.subheader("🔬 Experto de Rendimiento")
                st.metric("Recomienda:", verdicts["xg"])
            
            st.markdown("---")
            st.header("⭐ Veredicto final e indicador de confianza")

            votes = [v for v in verdicts.values() if v != "Datos insuficientes"]
            if len(votes) > 0:
                most_common_verdict = max(set(votes), key=votes.count)
                num_votes = votes.count(most_common_verdict)
                st.metric(f"El resultado más probable:", f"{most_common_verdict}", f"{num_votes} de {len(votes)} expertos están de acuerdo")
            else:
                st.warning("No se puede calcular el indicador de confianza debido a la falta de todos los análisis.")

        with tab2:
            st.header("Análisis del mercado de resultado del partido (1x2)")
            if not any(s.get('edge', 0) > 0 for s in sugg_h2h.values()):
                st.info("No hay oportunidades de valor claras en este mercado según los criterios actuales.")
            else:
                for side, suggestion in sugg_h2h.items():
                    if suggestion.get('edge', 0) > 0:
                        with st.container(border=True):
                            st.subheader(f"🎯 Oportunidad de valor: {side.capitalize()}")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Mejor cuota del mercado", f"{agg_odds_h2h.get(side, 0):.2f}")
                            c2.metric("Ventaja (Edge)", f"+{suggestion['edge']*100:.2f}%")
                            c3.metric("Apuesta sugerida (Kelly)", f"${suggestion['stake_amount']:.2f}")

        with tab3:
            st.header("Análisis del mercado de goles (Más/Menos)")
            totals_lines = odds_api.extract_totals_lines(event)
            if not totals_lines:
                st.info("No hay datos para el mercado de goles en este partido.")
            else:
                selected_line = st.selectbox("Seleccione la línea de gol:", sorted(totals_lines.keys(), key=float))
                line_data = totals_lines[selected_line]
                agg_odds_ou = {'over': aggregate_prices(line_data.get('over', []), 'best'), 'under': aggregate_prices(line_data.get('under', []), 'best')}
                if agg_odds_ou['over'] > 0 and agg_odds_ou['under'] > 0:
                    imps_ou = implied_from_decimal(agg_odds_ou)
                    fair_ou = shin_fair_probs(imps_ou)
                    sugg_ou = kelly_suggestions(fair_ou, agg_odds_ou, bankroll, kelly_scale)
                    st.subheader(f"Probabilidades justas para la línea {selected_line}")
                    st.markdown(render_prob_bar(f"Más de {selected_line}", fair_ou.get('over', 0), '#22c55e'), unsafe_allow_html=True)
                    st.markdown(render_prob_bar(f"Menos de {selected_line}", fair_ou.get('under', 0), '#ef4444'), unsafe_allow_html=True)
                    if any(s.get('edge', 0) > 0 for s in sugg_ou.values()):
                        for side, suggestion in sugg_ou.items():
                            if suggestion.get('edge', 0) > 0:
                                with st.container(border=True):
                                    st.subheader(f"🎯 Oportunidad de valor: {side.capitalize()} {selected_line}")
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Mejor cuota", f"{agg_odds_ou.get(side, 0):.2f}")
                                    c2.metric("Ventaja (Edge)", f"+{suggestion['edge']*100:.2f}%")
                                    c3.metric("Apuesta sugerida", f"${suggestion['stake_amount']:.2f}")
                else:
                    st.warning("No hay suficientes cuotas para analizar esta línea.")

        with tab4:
            if st.button("Pedir un análisis detallado al presidente del consejo 🧠"):
                if not gemini_api_key:
                    st.error("Por favor, ingrese primero la clave API de Gemini.")
                else:
                    with st.spinner("El estratega está pensando..."):
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
                            st.error(f"Error de Gemini: {e}")
