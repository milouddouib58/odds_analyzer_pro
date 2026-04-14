# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import json
import time

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import scipy.optimize as optimize

HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except:
    pass

# ============================ إعدادات الصفحة ============================
st.set_page_config(
    page_title="⚽ Pro Sports Betting Lab — Odds API",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ثوابت
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
EPS = 1e-9
RANDOM_STATE = 42

# ============================ CSS مخصص ============================
st.markdown("""
<style>
.big-metric {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    margin: 5px;
}
.value-bet {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 12px;
    padding: 15px;
    color: white;
    font-weight: bold;
}
.no-bet {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 12px;
    padding: 15px;
    color: white;
}
.warning-bet {
    background: linear-gradient(135deg, #f39c12, #f1c40f);
    border-radius: 12px;
    padding: 15px;
    color: #333;
}
.odds-card {
    border: 2px solid #2196F3;
    border-radius: 10px;
    padding: 15px;
    margin: 5px;
    background: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# ============================ دوال The Odds API ============================
class OddsAPIClient:
    """عميل متقدم لـ The Odds API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = ODDS_API_BASE
        self.remaining_requests = None
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        try:
            resp = self.session.get(
                f"{self.base_url}{endpoint}", params=params, timeout=15
            )
            self.remaining_requests = resp.headers.get('x-requests-remaining', 'N/A')
            if resp.status_code == 401:
                raise ValueError("❌ API Key غير صالح")
            elif resp.status_code == 422:
                raise ValueError("❌ معامل غير صالح")
            elif resp.status_code == 429:
                raise ValueError("❌ تجاوز حد الطلبات")
            elif resp.status_code != 200:
                raise ValueError(f"❌ خطأ API: {resp.status_code}")
            return resp.json()
        except requests.exceptions.Timeout:
            raise ValueError("❌ انتهت مهلة الاتصال")
        except requests.exceptions.ConnectionError:
            raise ValueError("❌ خطأ في الاتصال بالإنترنت")

    def get_sports(self) -> pd.DataFrame:
        data = self._get("/sports")
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_odds(self, sport: str, regions: str = "eu,uk,us",
                 markets: str = "h2h,spreads,totals",
                 odds_format: str = "decimal") -> list:
        return self._get(f"/sports/{sport}/odds", {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': 'iso'
        })

    def get_scores(self, sport: str, days_from: int = 3) -> list:
        return self._get(f"/sports/{sport}/scores", {
            'daysFrom': days_from
        })

    def get_historical_odds(self, sport: str, date: str) -> list:
        """جلب أوديدات تاريخية (يحتاج خطة مدفوعة)"""
        return self._get(f"/sports/{sport}/odds-history", {
            'date': date,
            'regions': 'eu',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        })


# ============================ معالجة بيانات الأوديدات ============================
class OddsProcessor:
    """معالج متقدم لبيانات الأوديدات - مستوحى من مكاتب المراهنات"""

    @staticmethod
    def decimal_to_probability(decimal_odds: float) -> float:
        """تحويل الأوديدات العشرية إلى احتمالات خام"""
        if decimal_odds <= 1.0:
            return 0.99
        return 1.0 / decimal_odds

    @staticmethod
    def remove_overround(probs: list) -> list:
        """
        إزالة هامش الربح (Overround/Vig) من الأوديدات
        طريقة Shin المستخدمة في المكاتب الاحترافية
        """
        total = sum(probs)
        if total <= 0:
            return probs
        # نرمليز مع الإزالة البسيطة أولاً
        normalized = [p / total for p in probs]
        return normalized

    @staticmethod
    def shin_method(decimal_odds_list: list) -> list:
        """
        طريقة Shin لإزالة Overround بدقة أعلى
        المستخدمة في التحليل الاحترافي
        """
        raw_probs = [1.0 / o for o in decimal_odds_list if o > 1]
        if not raw_probs:
            return []
        total = sum(raw_probs)
        if total <= 1.0:
            return raw_probs

        # حل معادلة Shin
        try:
            def shin_equation(z):
                n = len(raw_probs)
                result = sum(
                    np.sqrt(z**2 + 4 * (1 - z) * p**2 / total) for p in raw_probs
                ) - (2 - z) * n
                return result

            result = optimize.brentq(shin_equation, 0.001, 0.3)
            z = result
            true_probs = [
                (np.sqrt(z**2 + 4 * (1 - z) * p**2 / total) - z) / (2 * (1 - z))
                for p in raw_probs
            ]
            # نرمليز
            s = sum(true_probs)
            return [p / s for p in true_probs]
        except:
            # fallback
            return [p / total for p in raw_probs]

    @staticmethod
    def compute_market_efficiency(bookmaker_odds: dict) -> dict:
        """
        حساب كفاءة السوق عبر مقارنة الكتب
        النظرة المحترفة: انتشار الأوديدات = المعلومات
        """
        all_home = []
        all_draw = []
        all_away = []
        for bookie, odds in bookmaker_odds.items():
            if 'home' in odds:
                all_home.append(odds['home'])
            if 'draw' in odds:
                all_draw.append(odds['draw'])
            if 'away' in odds:
                all_away.append(odds['away'])

        metrics = {}
        for name, values in [('home', all_home), ('draw', all_draw), ('away', all_away)]:
            if len(values) >= 2:
                metrics[f'{name}_max'] = max(values)
                metrics[f'{name}_min'] = min(values)
                metrics[f'{name}_mean'] = np.mean(values)
                metrics[f'{name}_std'] = np.std(values)
                metrics[f'{name}_spread_pct'] = (max(values) - min(values)) / np.mean(values) * 100
            else:
                metrics[f'{name}_max'] = values[0] if values else 0
                metrics[f'{name}_min'] = values[0] if values else 0
                metrics[f'{name}_mean'] = values[0] if values else 0
                metrics[f'{name}_std'] = 0
                metrics[f'{name}_spread_pct'] = 0
        return metrics

    @staticmethod
    def find_value_bets(true_prob: float, best_decimal_odds: float,
                        min_edge: float = 0.02) -> dict:
        """
        كشف رهانات القيمة (Value Bets)
        القاعدة: EV = true_prob * (odds - 1) - (1 - true_prob) > edge
        """
        implied_prob = 1.0 / best_decimal_odds if best_decimal_odds > 1 else 0
        edge = true_prob - implied_prob
        ev = true_prob * (best_decimal_odds - 1) - (1 - true_prob)
        roi = ev / 1.0 * 100
        is_value = edge >= min_edge and ev > 0

        return {
            'true_prob': true_prob,
            'implied_prob': implied_prob,
            'edge': edge,
            'ev': ev,
            'roi_pct': roi,
            'is_value_bet': is_value,
            'best_odds': best_decimal_odds,
            'rating': 'STRONG VALUE ⭐⭐⭐' if edge >= 0.08 else 'VALUE ⭐⭐' if edge >= 0.05 else 'SLIGHT VALUE ⭐' if edge >= 0.02 else 'NO VALUE ❌'
        }

    @staticmethod
    def kelly_criterion(true_prob: float, decimal_odds: float,
                        bankroll: float = 1000, fraction: float = 0.25) -> dict:
        """
        Kelly Criterion لتحديد حجم الرهان المثلى
        fraction=0.25 -> Quarter Kelly (أكثر أماناً)
        """
        b = decimal_odds - 1
        p = true_prob
        q = 1 - p
        if b <= 0 or p <= 0:
            return {'kelly_pct': 0, 'bet_size': 0, 'recommended': False}

        full_kelly = (b * p - q) / b
        fractional_kelly = full_kelly * fraction
        fractional_kelly = max(0, min(fractional_kelly, 0.25))
        bet_size = bankroll * fractional_kelly

        return {
            'full_kelly_pct': full_kelly * 100,
            'quarter_kelly_pct': fractional_kelly * 100,
            'bet_size': bet_size,
            'recommended': full_kelly > 0 and fractional_kelly > 0.005,
            'risk_level': 'منخفض' if fractional_kelly < 0.05 else 'متوسط' if fractional_kelly < 0.10 else 'مرتفع'
        }

    @staticmethod
    def compute_closing_line_value(opening_odds: float, closing_odds: float) -> float:
        """
        Closing Line Value (CLV) - أهم مؤشر في المراهنات الاحترافية
        إذا كان CLV > 0 فأنت تتفوق على السوق
        """
        if opening_odds <= 1 or closing_odds <= 1:
            return 0.0
        opening_prob = 1.0 / opening_odds
        closing_prob = 1.0 / closing_odds
        return (closing_prob - opening_prob) / opening_prob * 100


# ============================ نموذج التنبؤ ============================
class MatchPredictor:
    """
    نموذج تنبؤ المباريات المتقدم
    يدمج عدة طرق: إحصائية + ML + سوق الأوديدات
    """

    def __init__(self):
        self.processor = OddsProcessor()

    def build_features_from_odds(self, match_data: dict,
                                 bookmaker_odds: dict) -> pd.DataFrame:
        """
        بناء ميزات من بيانات الأوديدات
        مستوحى من نماذج Sharp Bettors
        """
        features = {}

        # 1. الأوديدات المُحوّلة (أفضل أوديد من كل الكتب)
        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []
        for bookie, odds in bookmaker_odds.items():
            if odds.get('home', 0) > 1:
                home_odds_list.append(odds['home'])
            if odds.get('draw', 0) > 1:
                draw_odds_list.append(odds['draw'])
            if odds.get('away', 0) > 1:
                away_odds_list.append(odds['away'])

        if not home_odds_list or not away_odds_list:
            return pd.DataFrame()

        # أفضل الأوديدات (Pinnacle-like)
        features['best_home_odds'] = max(home_odds_list)
        features['best_away_odds'] = max(away_odds_list)
        features['avg_home_odds'] = np.mean(home_odds_list)
        features['avg_away_odds'] = np.mean(away_odds_list)

        # التعامل مع الأسواق الثنائية (بدون تعادل) والأسواق الثلاثية
        is_3way = len(draw_odds_list) > 0
        if is_3way:
            features['best_draw_odds'] = max(draw_odds_list)
            features['avg_draw_odds'] = np.mean(draw_odds_list)
            avg_odds = [features['avg_home_odds'], features['avg_draw_odds'], features['avg_away_odds']]
        else:
            features['best_draw_odds'] = 0.0
            features['avg_draw_odds'] = 0.0
            avg_odds = [features['avg_home_odds'], features['avg_away_odds']]

        # 2. الاحتمالات الحقيقية (Shin Method)
        shin_probs = self.processor.shin_method(avg_odds)
        features['shin_home_prob'] = shin_probs[0]
        if is_3way and len(shin_probs) >= 3:
            features['shin_draw_prob'] = shin_probs[1]
            features['shin_away_prob'] = shin_probs[2]
        else:
            features['shin_draw_prob'] = 0.0
            features['shin_away_prob'] = shin_probs[1] if len(shin_probs) > 1 else 0.5

        # 3. مقاييس اختلاف الكتب (إشارات السوق)
        efficiency = self.processor.compute_market_efficiency(bookmaker_odds)
        features.update({k: v for k, v in efficiency.items()})

        # 4. Overround (هامش ربح الكتاب)
        raw_probs_sum = (1 / features['avg_home_odds']) + (1 / features['avg_away_odds'])
        if is_3way:
            raw_probs_sum += (1 / features['avg_draw_odds'])
            
        features['overround'] = (raw_probs_sum - 1) * 100
        features['num_bookmakers'] = len(bookmaker_odds)

        # 5. نسب الأوديدات (مؤشرات ميزانية الفريقين)
        features['home_away_ratio'] = (
            features['avg_away_odds'] / features['avg_home_odds']
            if features['avg_home_odds'] > 0 else 1.0
        )
        features['home_prob_advantage'] = (
            features['shin_home_prob'] - features['shin_away_prob']
        )

        # 6. تشتت الأوديدات (كلما زاد كان السوق أقل يقيناً)
        features['home_odds_std'] = np.std(home_odds_list) if len(home_odds_list) > 1 else 0
        features['away_odds_std'] = np.std(away_odds_list) if len(away_odds_list) > 1 else 0

        # 7. ميزة الملعب (Home Advantage Factor)
        features['home_advantage_factor'] = max(
            0, features['shin_home_prob'] - features['shin_away_prob']
        )

        return pd.DataFrame([features])

    def predict_match(self, home_team: str, away_team: str,
                      bookmaker_odds: dict, historical_data: pd.DataFrame = None,
                      sport_key: str = 'soccer') -> dict:
        """
        التنبؤ الشامل للمباراة
        يجمع بين: الأوديدات + ML + إحصائيات تاريخية
        """
        X = self.build_features_from_odds(
            {'home': home_team, 'away': away_team},
            bookmaker_odds
        )
        if X.empty:
            return {'error': 'بيانات غير كافية'}

        # الاحتمالات من Shin Method
        shin_home = float(X['shin_home_prob'].iloc[0])
        shin_draw = float(X['shin_draw_prob'].iloc[0])
        shin_away = float(X['shin_away_prob'].iloc[0])

        # تصحيح ميزة الملعب (مرتبط بنوع الرياضة)
        home_advantage = 0.03 if 'soccer' in sport_key else 0.02
        adjusted_home = min(shin_home + home_advantage, 0.95)
        
        if shin_draw > 0:
            adjusted_away = max(shin_away - home_advantage * 0.5, 0.02)
            adjusted_draw = max(1 - adjusted_home - adjusted_away, 0.02)
        else:
            adjusted_away = max(1 - adjusted_home, 0.02)
            adjusted_draw = 0.0

        # تحليل القيمة
        best_home = float(X['best_home_odds'].iloc[0])
        best_draw = float(X['best_draw_odds'].iloc[0])
        best_away = float(X['best_away_odds'].iloc[0])

        home_value = self.processor.find_value_bets(adjusted_home, best_home)
        away_value = self.processor.find_value_bets(adjusted_away, best_away)
        draw_value = self.processor.find_value_bets(adjusted_draw, best_draw) if shin_draw > 0 else None

        # Kelly
        bankroll = 1000
        home_kelly = self.processor.kelly_criterion(adjusted_home, best_home, bankroll)
        away_kelly = self.processor.kelly_criterion(adjusted_away, best_away, bankroll)
        draw_kelly = self.processor.kelly_criterion(adjusted_draw, best_draw, bankroll) if shin_draw > 0 else None

        # أفضل رهان
        value_bets = []
        options = [
            ('الفوز للمضيف', home_value, home_kelly, best_home),
            ('الفوز للضيف', away_value, away_kelly, best_away)
        ]
        
        if shin_draw > 0:
            options.append(('التعادل', draw_value, draw_kelly, best_draw))

        for outcome, vb, kelly, odds in options:
            if vb and vb['is_value_bet']:
                value_bets.append({
                    'outcome': outcome,
                    'edge': vb['edge'],
                    'ev': vb['ev'],
                    'odds': odds,
                    'kelly_bet': kelly['bet_size'],
                    'rating': vb['rating']
                })

        value_bets.sort(key=lambda x: x['ev'], reverse=True)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'probs': {
                'home': adjusted_home,
                'draw': adjusted_draw,
                'away': adjusted_away
            },
            'shin_probs': {
                'home': shin_home,
                'draw': shin_draw,
                'away': shin_away
            },
            'best_odds': {
                'home': best_home,
                'draw': best_draw,
                'away': best_away
            },
            'value_analysis': {
                'home': home_value,
                'draw': draw_value,
                'away': away_value
            },
            'kelly': {
                'home': home_kelly,
                'draw': draw_kelly,
                'away': away_kelly
            },
            'value_bets': value_bets,
            'market_info': {
                'overround': float(X['overround'].iloc[0]),
                'num_bookmakers': int(X['num_bookmakers'].iloc[0]),
                'home_odds_spread': float(X.get('home_spread_pct', pd.Series([0])).iloc[0]
                                          if 'home_spread_pct' in X.columns else 0)
            },
            'recommendation': value_bets[0] if value_bets else None
        }


# ============================ دوال مساعدة للعرض ============================

def parse_odds_data(raw_odds_data: list) -> list:
    """تحليل بيانات الأوديدات الخام"""
    matches = []
    for event in raw_odds_data:
        try:
            match_info = {
                'id': event.get('id', ''),
                'sport': event.get('sport_key', ''),
                'home_team': event.get('home_team', ''),
                'away_team': event.get('away_team', ''),
                'commence_time': event.get('commence_time', ''),
                'bookmakers': {}
            }
            for bookie in event.get('bookmakers', []):
                bookie_name = bookie.get('title', '')
                bookie_odds = {}
                for market in bookie.get('markets', []):
                    market_key = market.get('key', '')
                    if market_key == 'h2h':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            name = outcome.get('name', '')
                            price = outcome.get('price', 0)
                            if name == event.get('home_team', ''):
                                bookie_odds['home'] = float(price)
                            elif name == event.get('away_team', ''):
                                bookie_odds['away'] = float(price)
                            elif name == 'Draw':
                                bookie_odds['draw'] = float(price)
                    elif market_key == 'spreads':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('name') == event.get('home_team'):
                                bookie_odds['spread_home'] = outcome.get('point', 0)
                                bookie_odds['spread_home_odds'] = outcome.get('price', 0)
                    elif market_key == 'totals':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('name') == 'Over':
                                bookie_odds['total_line'] = outcome.get('point', 0)
                                bookie_odds['over_odds'] = outcome.get('price', 0)
                            elif outcome.get('name') == 'Under':
                                bookie_odds['under_odds'] = outcome.get('price', 0)
                if bookie_odds:
                    match_info['bookmakers'][bookie_name] = bookie_odds
            if match_info['bookmakers']:
                matches.append(match_info)
        except Exception as e:
            continue
    return matches


def format_match_time(iso_time: str) -> str:
    """تنسيق وقت المباراة"""
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        dt_local = dt.astimezone(pytz.timezone('Asia/Riyadh'))
        return dt_local.strftime('%Y-%m-%d %H:%M')
    except:
        return iso_time


def compute_expected_value_match(prob: float, decimal_odds: float) -> float:
    return prob * (decimal_odds - 1) - (1 - prob)


def create_odds_comparison_chart(bookmaker_odds: dict, home_team: str,
                                 away_team: str) -> go.Figure:
    """مخطط مقارنة الأوديدات عبر الكتب"""
    bookies = []
    home_odds_list = []
    draw_odds_list = []
    away_odds_list = []

    for bookie, odds in bookmaker_odds.items():
        if 'home' in odds and 'away' in odds:
            bookies.append(bookie)
            home_odds_list.append(odds.get('home', 0))
            draw_odds_list.append(odds.get('draw', 0))
            away_odds_list.append(odds.get('away', 0))

    if not bookies:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f'🏠 {home_team}',
        x=bookies,
        y=home_odds_list,
        marker_color='#2196F3',
        text=[f"{o:.2f}" for o in home_odds_list],
        textposition='outside'
    ))

    if any(d > 0 for d in draw_odds_list):
        fig.add_trace(go.Bar(
            name='🤝 تعادل',
            x=bookies,
            y=draw_odds_list,
            marker_color='#FF9800',
            text=[f"{o:.2f}" for o in draw_odds_list],
            textposition='outside'
        ))

    fig.add_trace(go.Bar(
        name=f'✈️ {away_team}',
        x=bookies,
        y=away_odds_list,
        marker_color='#F44336',
        text=[f"{o:.2f}" for o in away_odds_list],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"مقارنة الأوديدات: {home_team} vs {away_team}",
        barmode='group',
        xaxis_title="الكتاب",
        yaxis_title="الأوديد العشري",
        legend_title="النتيجة",
        height=400
    )
    return fig


def create_probability_gauge(prob: float, title: str) -> go.Figure:
    """مقياس الاحتمال"""
    color = ('#2ECC71' if prob >= 0.6 else '#F39C12' if prob >= 0.4 else '#E74C3C')
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': '#FFEBEE'},
                {'range': [33, 60], 'color': '#FFF9C4'},
                {'range': [60, 100], 'color': '#E8F5E9'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def display_value_bet_card(outcome: str, prob: float, odds: float,
                           value_info: dict, kelly_info: dict):
    """عرض بطاقة رهان القيمة"""
    if not value_info: return
    edge = value_info['edge']
    ev = value_info['ev']
    is_value = value_info['is_value_bet']

    if is_value and edge >= 0.05:
        css_class = "value-bet"
        icon = "⭐"
    elif is_value:
        css_class = "warning-bet"
        icon = "💛"
    else:
        css_class = "no-bet"
        icon = "❌"

    st.markdown(f"""
    <div class="{css_class}">
        <h4>{icon} {outcome}</h4>
        <p>الاحتمال الحقيقي: <strong>{prob*100:.1f}%</strong> | الأوديد: <strong>{odds:.2f}</strong></p>
        <p>الحافة: <strong>{edge*100:+.1f}%</strong> | القيمة المتوقعة: <strong>{ev:+.3f}</strong></p>
        <p>التقييم: <strong>{value_info['rating']}</strong></p>
        <p>رهان Kelly: <strong>{kelly_info.get('bet_size', 0):.0f} وحدة</strong> (خطر: {kelly_info.get('risk_level', 'N/A')})</p>
    </div>
    """, unsafe_allow_html=True)


def _display_full_analysis(result: dict, match: dict, bankroll: float,
                           kelly_fraction: float, min_edge: float):
    """عرض التحليل الكامل للمباراة"""
    home = result['home_team']
    away = result['away_team']
    probs = result['probs']
    best_odds = result['best_odds']

    st.markdown(f"## ⚽ {home} vs {away}")

    has_draw = probs.get('draw', 0) > 0

    # مقاييس الاحتمالات
    pcols = st.columns(3 if has_draw else 2)
    with pcols[0]:
        fig = create_probability_gauge(probs['home'], f"🏠 {home}")
        st.plotly_chart(fig, use_container_width=True, key=f"prob_home_{home}_{away}")
    if has_draw and len(pcols) == 3:
        with pcols[1]:
            fig = create_probability_gauge(probs['draw'], "🤝 تعادل")
            st.plotly_chart(fig, use_container_width=True, key=f"prob_draw_{home}_{away}")
    with pcols[-1]:
        fig = create_probability_gauge(probs['away'], f"✈️ {away}")
        st.plotly_chart(fig, use_container_width=True, key=f"prob_away_{home}_{away}")

    # مخطط مقارنة الأوديدات
    fig_compare = create_odds_comparison_chart(
        match['bookmakers'], home, away
    )
    st.plotly_chart(fig_compare, use_container_width=True, key=f"compare_{home}_{away}")

    # تحليل القيمة
    st.subheader("💎 تحليل القيمة (Value Analysis)")
    vcols = st.columns(3 if has_draw else 2)

    with vcols[0]:
        va = result['value_analysis']
        display_value_bet_card(
            f"🏠 {home}",
            probs['home'],
            best_odds['home'],
            va['home'],
            result['kelly']['home']
        )
    if has_draw and len(vcols) == 3:
        with vcols[1]:
            display_value_bet_card(
                "🤝 تعادل",
                probs['draw'],
                best_odds.get('draw', 3.5),
                va['draw'],
                result['kelly']['draw']
            )
    with vcols[-1]:
        display_value_bet_card(
            f"✈️ {away}",
            probs['away'],
            best_odds['away'],
            va['away'],
            result['kelly']['away']
        )

    # التوصية النهائية
    st.subheader("🎯 التوصية النهائية")
    if result['recommendation']:
        rec = result['recommendation']
        st.markdown(f"""
        <div class="value-bet" style="font-size:18px; padding:25px;">
            <h3>✅ راهن على: {rec['outcome']}</h3>
            <p>📊 الأوديد: <strong>{rec['odds']:.2f}</strong> | 💰 حجم الرهان: <strong>{rec['kelly_bet']:.0f} وحدة</strong></p>
            <p>📈 الحافة: <strong>{rec['edge']*100:+.1f}%</strong> | 💎 القيمة المتوقعة: <strong>{rec['ev']:+.3f}</strong></p>
            <p>⭐ التقييم: <strong>{rec['rating']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-bet" style="padding:25px; font-size:16px;">
            <h3>❌ لا توجد قيمة حقيقية في هذه المباراة</h3>
            <p>الأوديدات لا توفر حافة إيجابية بعد إزالة Overround</p>
            <p>تخطى هذه المباراة واحفظ رأس مالك</p>
        </div>
        """, unsafe_allow_html=True)

    # معلومات السوق
    st.subheader("📊 معلومات السوق")
    mi = result['market_info']
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Overround (هامش الكتاب)", f"{mi['overround']:.2f}%",
               delta="مرتفع ❌" if mi['overround'] > 7 else "معقول ✅",
               delta_color="inverse")
    mc2.metric("عدد الكتب", mi['num_bookmakers'])
    mc3.metric("أفضل رهان", result['value_bets'][0]['outcome'] if result['value_bets'] else "لا يوجد")


# ============================ الشريط الجانبي ============================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/football.png", width=80)
    st.title("⚽ Pro Betting Lab")
    st.markdown("---")

    # API Key
    st.subheader("🔑 إعدادات API")
    api_key = st.text_input(
        "The Odds API Key",
        type="password",
        placeholder="أدخل API Key هنا...",
        help="احصل على مفتاح مجاني من: the-odds-api.com"
    )
    if api_key:
        st.success("✅ API Key مُدخل", icon="🔑")
    else:
        st.info("💡 ستحتاج API Key للحصول على بيانات حية", icon="ℹ️")
    st.markdown("[🔗 احصل على API Key مجاني](https://the-odds-api.com/)")
    st.markdown("---")

    # إعدادات الرهان
    st.subheader("💰 إعدادات إدارة رأس المال")
    bankroll = st.number_input("رأس المال (وحدة)", value=1000.0, step=100.0)
    kelly_fraction = st.slider("كسر Kelly", 0.1, 1.0, 0.25, 0.05,
                               help="0.25 = Quarter Kelly (الأكثر أماناً)")
    min_edge = st.slider("الحافة الدنيا للرهان", 0.01, 0.15, 0.03, 0.01,
                         help="الفرق الدني بين احتمالك وضمني الأوديد")
    min_odds = st.slider("الأوديد الأدنى للرهان", 1.30, 3.00, 1.50, 0.05)
    max_odds = st.slider("الأوديد الأقصى للرهان", 2.0, 20.0, 8.0, 0.5)
    st.markdown("---")

    # إعدادات السوق
    st.subheader("📊 إعدادات السوق")
    regions = st.multiselect(
        "المناطق",
        ['eu', 'uk', 'us', 'au'],
        default=['eu', 'uk'],
        help="المناطق الجغرافية للكتب"
    )
    markets = st.multiselect(
        "الأسواق",
        ['h2h', 'spreads', 'totals'],
        default=['h2h'],
        help="أنواع الأسواق المطلوبة"
    )
    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================ الواجهة الرئيسية ============================
st.title("⚽ Pro Sports Betting Lab — Powered by The Odds API")
st.caption(
    "نظام تحليل رياضي احترافي | Shin Method + Value Betting + Kelly Criterion + Market Analysis"
)

# تحذير
st.warning(
    "⚠️ هذا النظام للأغراض التعليمية والترفيهية فقط. "
    "المراهنات تنطوي على مخاطر مالية عالية. "
    "لا تراهن بأكثر مما تستطيع خسارته.",
    icon="⚠️"
)

# ============================ التبويبات ============================
tabs = st.tabs([
    "🏠 لوحة التحكم",
    "🔴 المباريات الحية",
    "📊 تحليل مباراة",
    "💎 Value Bets",
    "📈 إدارة رأس المال",
    "🧮 حاسبة الأوديدات"
])

# ========================= لوحة التحكم =========================
with tabs[0]:
    st.header("🏠 لوحة التحكم الرئيسية")

    if not api_key:
        st.info("👆 أدخل API Key في الشريط الجانبي للبدء", icon="🔑")
        # عرض ديمو
        st.subheader("📖 كيف يعمل النظام؟")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="big-metric">
                <h2>🎯</h2>
                <h4>Shin Method</h4>
                <p>إزالة Overround بدقة عالية</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="big-metric">
                <h2>💎</h2>
                <h4>Value Betting</h4>
                <p>كشف الرهانات ذات القيمة الحقيقية</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="big-metric">
                <h2>📊</h2>
                <h4>Kelly Criterion</h4>
                <p>إدارة رأس المال رياضياً</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="big-metric">
                <h2>📉</h2>
                <h4>CLV Analysis</h4>
                <p>تحليل Closing Line Value</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🏆 الرياضات المدعومة")
        sports_demo = [
            {'الرياضة': 'كرة القدم', 'الدوريات': 'الدوري الإنجليزي، الإسباني، الألماني، الفرنسي...', 'المفتاح': 'soccer_*'},
            {'الرياضة': 'كرة السلة', 'الدوريات': 'NBA، EuroLeague', 'المفتاح': 'basketball_*'},
            {'الرياضة': 'كرة القدم الأمريكية', 'الدوريات': 'NFL، NCAAF', 'المفتاح': 'americanfootball_*'},
            {'الرياضة': 'التنس', 'الدوريات': 'ATP، WTA، Grand Slams', 'المفتاح': 'tennis_*'},
            {'الرياضة': 'الهوكي', 'الدوريات': 'NHL', 'المفتاح': 'icehockey_*'},
            {'الرياضة': 'البيسبول', 'الدوريات': 'MLB', 'المفتاح': 'baseball_*'},
        ]
        st.dataframe(pd.DataFrame(sports_demo), use_container_width=True)

    else:
        client = OddsAPIClient(api_key)
        with st.spinner("جلب الرياضات المتاحة..."):
            try:
                sports_df = client.get_sports()
                if not sports_df.empty:
                    active_sports = sports_df[sports_df.get('active', pd.Series(True)) == True] if 'active' in sports_df.columns else sports_df
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("إجمالي الرياضات", len(sports_df))
                    c2.metric("الرياضات النشطة", len(active_sports))
                    c3.metric("طلبات متبقية", client.remaining_requests)
                    c4.metric("آخر تحديث", datetime.now().strftime("%H:%M:%S"))

                    st.subheader("الرياضات المتاحة")
                    # تصفية وعرض
                    if 'group' in sports_df.columns:
                        groups = sports_df['group'].unique().tolist()
                        selected_group = st.selectbox("فئة الرياضة", ['الكل'] + groups)
                        if selected_group != 'الكل':
                            filtered = sports_df[sports_df['group'] == selected_group]
                        else:
                            filtered = sports_df
                    else:
                        filtered = sports_df

                    display_cols = ['key', 'title', 'group', 'active'] if 'group' in filtered.columns else filtered.columns[:4]
                    st.dataframe(
                        filtered[[c for c in display_cols if c in filtered.columns]],
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.warning("لا توجد رياضات متاحة")
            except Exception as e:
                st.error(f"خطأ: {e}")

# ========================= المباريات الحية =========================
with tabs[1]:
    st.header("🔴 المباريات القادمة والحية")

    if not api_key:
        st.info("أدخل API Key للوصول إلى البيانات الحية", icon="🔑")
    else:
        client = OddsAPIClient(api_key)

        # اختيار الرياضة
        try:
            sports_df = client.get_sports()
            sport_options = {}
            if not sports_df.empty and 'key' in sports_df.columns:
                for _, row in sports_df.iterrows():
                    title = row.get('title', row['key'])
                    sport_options[f"{title}"] = row['key']
            else:
                sport_options = {
                    'كرة القدم - Premier League': 'soccer_epl',
                    'كرة القدم - La Liga': 'soccer_spain_la_liga',
                    'كرة القدم - Champions League': 'soccer_uefa_champs_league',
                    'كرة السلة - NBA': 'basketball_nba',
                    'NFL': 'americanfootball_nfl',
                }
        except:
            sport_options = {
                'كرة القدم - Premier League': 'soccer_epl',
                'كرة القدم - La Liga': 'soccer_spain_la_liga',
                'كرة السلة - NBA': 'basketball_nba',
            }

        col_s1, col_s2, col_s3 = st.columns(3)
        selected_sport_name = col_s1.selectbox("الرياضة", list(sport_options.keys()))
        selected_sport_key = sport_options[selected_sport_name]

        regions_str = ','.join(regions) if regions else 'eu'
        markets_str = ','.join(markets) if markets else 'h2h'
        auto_refresh = col_s2.checkbox("تحديث تلقائي كل دقيقة", value=False)
        show_all = col_s3.checkbox("عرض كل المباريات", value=True)

        if st.button("🔄 جلب المباريات", type="primary") or auto_refresh:
            with st.spinner(f"جلب مباريات {selected_sport_name}..."):
                try:
                    raw_odds = client.get_odds(
                        selected_sport_key,
                        regions=regions_str,
                        markets=markets_str
                    )
                    if not raw_odds:
                        st.warning("لا توجد مباريات متاحة الآن لهذه الرياضة")
                    else:
                        matches = parse_odds_data(raw_odds)
                        st.success(f"✅ تم جلب {len(matches)} مباراة | طلبات متبقية: {client.remaining_requests}")

                        # حفظ في session state
                        st.session_state['matches'] = matches
                        st.session_state['sport_key'] = selected_sport_key

                        # عرض المباريات
                        predictor = MatchPredictor()
                        for i, match in enumerate(matches):
                            home = match['home_team']
                            away = match['away_team']
                            match_time = format_match_time(match['commence_time'])
                            bookies = match['bookmakers']

                            with st.expander(
                                f"⚽ {home} vs {away} | 🕐 {match_time} | 📚 {len(bookies)} كتاب",
                                expanded=(i < 3)
                            ):
                                result = predictor.predict_match(
                                    home, away, bookies,
                                    sport_key=selected_sport_key
                                )

                                if 'error' not in result:
                                    c1, c2, c3, c4 = st.columns(4)
                                    probs = result['probs']
                                    best_odds = result['best_odds']

                                    c1.metric(
                                        f"🏠 {home}",
                                        f"{probs['home']*100:.1f}%",
                                        f"أوديد: {best_odds['home']:.2f}"
                                    )
                                    if probs.get('draw', 0) > 0:
                                        c2.metric(
                                            "🤝 تعادل",
                                            f"{probs['draw']*100:.1f}%",
                                            f"أوديد: {best_odds.get('draw', 0):.2f}"
                                        )
                                    else:
                                        c2.metric("🤝 تعادل", "N/A", "N/A")
                                        
                                    c3.metric(
                                        f"✈️ {away}",
                                        f"{probs['away']*100:.1f}%",
                                        f"أوديد: {best_odds['away']:.2f}"
                                    )

                                    # أفضل توصية
                                    if result['value_bets']:
                                        best = result['value_bets'][0]
                                        c4.markdown(f"""
                                        <div class="value-bet">
                                            <b>💎 Value Bet</b><br>
                                            {best['outcome']}<br>
                                            حافة: {best['edge']*100:+.1f}%
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        c4.markdown("""
                                        <div class="no-bet">
                                            <b>❌ لا توجد قيمة</b>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    # مقارنة الأوديدات
                                    fig_compare = create_odds_comparison_chart(bookies, home, away)
                                    st.plotly_chart(fig_compare, use_container_width=True, key=f"comp_live_{i}_{home}_{away}")

                                    # معلومات السوق
                                    mi = result['market_info']
                                    st.caption(
                                        f"📊 Overround: {mi['overround']:.2f}% | "
                                        f"عدد الكتب: {mi['num_bookmakers']}"
                                    )
                except Exception as e:
                    st.error(f"خطأ في جلب البيانات: {e}")

# ========================= تحليل مباراة =========================
with tabs[2]:
    st.header("📊 تحليل مباراة متعمق")

    if not api_key:
        st.info("أدخل API Key أولاً", icon="🔑")
    else:
        # اختيار من المباريات المحملة أو إدخال يدوي
        input_mode = st.radio(
            "طريقة الإدخال",
            ["من المباريات المحملة", "إدخال يدوي للأوديدات"],
            horizontal=True
        )

        if input_mode == "من المباريات المحملة":
            if 'matches' not in st.session_state:
                st.warning("اذهب إلى تبويب 'المباريات الحية' وجلّب المباريات أولاً")
            else:
                matches = st.session_state['matches']
                match_options = {
                    f"{m['home_team']} vs {m['away_team']}": i for i, m in enumerate(matches)
                }
                selected_match_name = st.selectbox("اختر المباراة", list(match_options.keys()))
                selected_idx = match_options[selected_match_name]
                match = matches[selected_idx]

                predictor = MatchPredictor()
                result = predictor.predict_match(
                    match['home_team'], match['away_team'], match['bookmakers'],
                    sport_key=st.session_state.get('sport_key', 'soccer')
                )

                if 'error' not in result:
                    _display_full_analysis(result, match, bankroll, kelly_fraction, min_edge)

        else:
            # إدخال يدوي
            st.subheader("إدخال الأوديدات يدوياً")
            col1, col2 = st.columns(2)
            home_team = col1.text_input("اسم الفريق المضيف", "Real Madrid")
            away_team = col2.text_input("اسم الفريق الضيف", "Barcelona")
            has_draw_input = st.checkbox("يوجد نتيجة تعادل (مثل كرة القدم)", value=True)

            st.subheader("الأوديدات العشرية من الكتب")
            bookie_data = {}
            num_bookies = st.slider("عدد الكتب", 1, 8, 4)
            bookie_names = [
                "Bet365", "William Hill", "Pinnacle", "Betfair",
                "Unibet", "888sport", "Ladbrokes", "Paddy Power"
            ]

            for i in range(num_bookies):
                bname = bookie_names[i] if i < len(bookie_names) else f"كتاب {i+1}"
                if has_draw_input:
                    cols = st.columns([2, 1, 1, 1])
                    bname_input = cols[0].text_input(f"اسم الكتاب {i+1}", bname, key=f"bn_{i}")
                    home_o = cols[1].number_input(f"🏠 {home_team}", 1.01, 50.0, 2.0, 0.01, key=f"ho_{i}")
                    draw_o = cols[2].number_input("🤝 تعادل", 1.01, 50.0, 3.5, 0.01, key=f"do_{i}")
                    away_o = cols[3].number_input(f"✈️ {away_team}", 1.01, 50.0, 3.5, 0.01, key=f"ao_{i}")
                    bookie_data[bname_input] = {'home': home_o, 'draw': draw_o, 'away': away_o}
                else:
                    cols = st.columns([2, 1, 1])
                    bname_input = cols[0].text_input(f"اسم الكتاب {i+1}", bname, key=f"bn2_{i}")
                    home_o = cols[1].number_input(f"🏠 {home_team}", 1.01, 50.0, 1.85, 0.01, key=f"ho2_{i}")
                    away_o = cols[2].number_input(f"✈️ {away_team}", 1.01, 50.0, 1.95, 0.01, key=f"ao2_{i}")
                    bookie_data[bname_input] = {'home': home_o, 'away': away_o}

            if st.button("🔍 تحليل المباراة", type="primary"):
                predictor = MatchPredictor()
                result = predictor.predict_match(
                    home_team, away_team, bookie_data, sport_key='soccer' if has_draw_input else 'tennis'
                )
                match = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'bookmakers': bookie_data,
                    'commence_time': datetime.now().isoformat()
                }

                if 'error' not in result:
                    _display_full_analysis(result, match, bankroll, kelly_fraction, min_edge)


# ========================= Value Bets =========================
with tabs[3]:
    st.header("💎 مسح Value Bets - جميع المباريات")

    if not api_key:
        st.info("أدخل API Key أولاً", icon="🔑")
    elif 'matches' not in st.session_state:
        st.warning("اذهب إلى 'المباريات الحية' وجلّب المباريات أولاً")
    else:
        matches = st.session_state['matches']
        predictor = MatchPredictor()

        all_value_bets = []
        with st.spinner("مسح جميع المباريات بحثاً عن قيمة..."):
            for match in matches:
                result = predictor.predict_match(
                    match['home_team'], match['away_team'], match['bookmakers'],
                    sport_key=st.session_state.get('sport_key', 'soccer')
                )
                if 'error' not in result:
                    match_time = format_match_time(match.get('commence_time', ''))
                    for vb in result['value_bets']:
                        if (vb['edge'] >= min_edge and
                            min_odds <= vb['odds'] <= max_odds):
                            all_value_bets.append({
                                'المباراة': f"{match['home_team']} vs {match['away_team']}",
                                'الوقت': match_time,
                                'الرهان': vb['outcome'],
                                'الأوديد': vb['odds'],
                                'الحافة %': f"{vb['edge']*100:+.1f}%",
                                'القيمة المتوقعة': f"{vb['ev']:+.3f}",
                                'التقييم': vb['rating'],
                                'حجم الرهان (Kelly)': f"{vb['kelly_bet']:.0f}",
                                '_edge': vb['edge'],
                                '_ev': vb['ev']
                            })

        if all_value_bets:
            vb_df = pd.DataFrame(all_value_bets)
            vb_df = vb_df.sort_values('_ev', ascending=False)
            st.success(f"✅ تم اكتشاف {len(all_value_bets)} رهان بقيمة إيجابية!")

            # إحصاءات
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("إجمالي Value Bets", len(all_value_bets))
            c2.metric("متوسط الحافة", f"{np.mean([v['_edge'] for v in all_value_bets])*100:.1f}%")
            c3.metric("أعلى حافة", f"{max(v['_edge'] for v in all_value_bets)*100:.1f}%")
            c4.metric("مجموع Kelly", f"{sum(float(v['حجم الرهان (Kelly)']) for v in all_value_bets):.0f}")

            # عرض الجدول
            display_df = vb_df.drop(columns=['_edge', '_ev'])
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )

            # مخطط الحواف
            fig_edges = px.bar(
                vb_df.head(20),
                x='المباراة',
                y='_edge',
                color='_ev',
                title="أفضل 20 Value Bet حسب الحافة",
                labels={'_edge': 'الحافة', '_ev': 'القيمة المتوقعة'},
                color_continuous_scale='RdYlGn'
            )
            fig_edges.update_xaxes(tickangle=45)
            st.plotly_chart(fig_edges, use_container_width=True, key="fig_edges_all")

            # تصدير
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "⬇️ تنزيل Value Bets (CSV)",
                csv,
                f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        else:
            st.warning("لم يتم اكتشاف value bets تلبي المعايير المحددة في هذه المباريات")

# ========================= إدارة رأس المال =========================
with tabs[4]:
    st.header("📈 إدارة رأس المال الاحترافية")

    st.subheader("🧮 حاسبة Kelly Criterion")
    col1, col2 = st.columns(2)
    with col1:
        kc_prob = st.slider("احتمالك للفوز (%)", 1, 99, 55, 1) / 100
        kc_odds = st.number_input("الأوديد العشري", 1.01, 50.0, 2.10, 0.01, key="kc_odds")
        kc_bankroll = st.number_input("رأس المال", 100.0, 100000.0, bankroll, 100.0)
        kc_fraction = st.slider("كسر Kelly", 0.1, 1.0, kelly_fraction, 0.05, key="kc_frac")
    with col2:
        processor = OddsProcessor()
        kelly_result = processor.kelly_criterion(kc_prob, kc_odds, kc_bankroll, kc_fraction)
        ev = compute_expected_value_match(kc_prob, kc_odds)
        st.markdown(f"""
        ### نتائج Kelly Criterion:
        - **Full Kelly:** {kelly_result['full_kelly_pct']:.2f}%
        - **Quarter Kelly ({kc_fraction*100:.0f}%):** {kelly_result['quarter_kelly_pct']:.2f}%
        - **حجم الرهان الموصى:** {kelly_result['bet_size']:.2f} وحدة
        - **القيمة المتوقعة:** {ev:+.4f}
        - **مستوى الخطر:** {kelly_result['risk_level']}
        """)
        if kelly_result['recommended']:
            st.success("✅ Kelly يوصي بالرهان", icon="✅")
        else:
            st.error("❌ Kelly لا يوصي بالرهان", icon="❌")

    st.markdown("---")

    # محاكاة نمو رأس المال
    st.subheader("📊 محاكاة نمو رأس المال (Monte Carlo)")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_bets = sim_col1.slider("عدد الرهانات", 10, 500, 100)
    sim_win_rate = sim_col2.slider("معدل الفوز (%)", 30, 70, 55) / 100
    sim_avg_odds = sim_col3.number_input("متوسط الأوديد", 1.5, 5.0, 2.0, 0.1)
    sim_kelly = st.slider("نسبة Kelly للمحاكاة", 0.05, 0.5, 0.25, 0.05)

    if st.button("🎲 تشغيل المحاكاة (1000 سيناريو)"):
        np.random.seed(RANDOM_STATE)
        n_sims = 1000
        initial_bankroll = bankroll
        all_paths = []
        final_bankrolls = []

        for sim in range(n_sims):
            br = initial_bankroll
            path = [br]
            for _ in range(sim_bets):
                kelly_bet_pct = (sim_avg_odds * sim_win_rate - (1 - sim_win_rate)) / (sim_avg_odds - 1)
                kelly_bet_pct = max(0, kelly_bet_pct) * sim_kelly
                bet_amount = br * kelly_bet_pct
                if np.random.random() < sim_win_rate:
                    br += bet_amount * (sim_avg_odds - 1)
                else:
                    br -= bet_amount
                br = max(br, 1)
                path.append(br)
            all_paths.append(path)
            final_bankrolls.append(br)

        # عرض النتائج
        final_arr = np.array(final_bankrolls)
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("متوسط النتيجة", f"{np.mean(final_arr):.0f}")
        rc2.metric("الوسيط", f"{np.median(final_arr):.0f}")
        rc3.metric("الأسوأ 10%", f"{np.percentile(final_arr, 10):.0f}")
        rc4.metric("الأفضل 10%", f"{np.percentile(final_arr, 90):.0f}")
        rc5.metric("احتمال الربح", f"{(final_arr > initial_bankroll).mean()*100:.0f}%")

        # مخطط مسارات المحاكاة
        fig_sim = go.Figure()
        # عرض 50 مسار عشوائي
        for path in all_paths[:50]:
            fig_sim.add_trace(go.Scatter(
                y=path,
                mode='lines',
                line=dict(width=0.5, color='rgba(100,150,200,0.3)'),
                showlegend=False
            ))
        # المتوسط
        avg_path = np.mean(all_paths, axis=0)
        fig_sim.add_trace(go.Scatter(
            y=avg_path,
            mode='lines',
            line=dict(width=3, color='green'),
            name='متوسط'
        ))
        fig_sim.add_hline(y=initial_bankroll, line_dash="dash", line_color="red",
                          annotation_text="رأس المال الأولي")
        fig_sim.update_layout(
            title=f"Monte Carlo: {n_sims} سيناريو × {sim_bets} رهان",
            xaxis_title="الرهان",
            yaxis_title="رأس المال"
        )
        st.plotly_chart(fig_sim, use_container_width=True, key="fig_sim_mc")

        # توزيع النتائج
        fig_dist = px.histogram(
            x=final_arr,
            nbins=50,
            title="توزيع النتائج النهائية",
            labels={'x': 'رأس المال النهائي', 'y': 'التكرار'}
        )
        fig_dist.add_vline(x=initial_bankroll, line_dash="dash", line_color="red",
                           annotation_text="أولي")
        st.plotly_chart(fig_dist, use_container_width=True, key="fig_dist_mc")


# ========================= حاسبة الأوديدات =========================
with tabs[5]:
    st.header("🧮 حاسبة الأوديدات الشاملة")

    calc_type = st.selectbox(
        "نوع الحساب",
        ["تحويل الأوديدات", "حساب Overround", "مقارنة الكتب", "حساب CLV"]
    )

    if calc_type == "تحويل الأوديدات":
        st.subheader("🔄 تحويل بين أنواع الأوديدات")
        input_type = st.radio("من:", ["عشري", "أمريكي", "كسري", "احتمال %"], horizontal=True)

        if input_type == "عشري":
            decimal_in = st.number_input("الأوديد العشري", 1.01, 100.0, 2.0, 0.01)
            prob = 1 / decimal_in
            american = int((decimal_in - 1) * 100) if decimal_in >= 2 else int(-100 / (decimal_in - 1))
            from fractions import Fraction
            frac = Fraction(int(prob * 1000), 1000).limit_denominator(100)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("عشري", f"{decimal_in:.2f}")
            c2.metric("أمريكي", f"+{american}" if american > 0 else str(american))
            c3.metric("كسري", f"{frac.denominator}/{frac.numerator}")
            c4.metric("احتمال", f"{prob*100:.2f}%")

        elif input_type == "احتمال %":
            prob_in = st.slider("الاحتمال (%)", 1, 99, 50) / 100
            decimal = 1 / prob_in
            american = int((decimal - 1) * 100) if decimal >= 2 else int(-100 / (decimal - 1))

            c1, c2, c3 = st.columns(3)
            c1.metric("عشري", f"{decimal:.3f}")
            c2.metric("أمريكي", f"+{american}" if american > 0 else str(american))
            c3.metric("احتمال", f"{prob_in*100:.1f}%")

    elif calc_type == "حساب Overround":
        st.subheader("📊 حساب Overround (هامش الكتاب)")
        num_outcomes = st.radio("عدد النتائج", [2, 3], horizontal=True)
        odds_inputs = []
        cols = st.columns(num_outcomes)
        labels = ['الفريق الأول', 'تعادل', 'الفريق الثاني']
        for i in range(num_outcomes):
            o = cols[i].number_input(labels[i], 1.01, 50.0, 2.0 if i != 1 else 3.5, 0.01, key=f"or_{i}")
            odds_inputs.append(o)

        raw_probs = [1 / o for o in odds_inputs]
        overround = (sum(raw_probs) - 1) * 100

        processor = OddsProcessor()
        shin_probs = processor.shin_method(odds_inputs)

        st.metric("Overround", f"{overround:.3f}%",
                  delta="مرتفع ❌" if overround > 7 else "معقول ✅",
                  delta_color="inverse")

        result_data = []
        out_labels = ['الأول', 'تعادل', 'الثاني'][:num_outcomes]
        for i, (label, raw, shin) in enumerate(zip(out_labels, raw_probs, shin_probs)):
            result_data.append({
                'النتيجة': label,
                'الأوديد': odds_inputs[i],
                'الاحتمال الخام': f"{raw*100:.2f}%",
                'الاحتمال (Shin)': f"{shin*100:.2f}%",
                'الفرق': f"{(shin-raw)*100:+.2f}%"
            })
        st.dataframe(pd.DataFrame(result_data), use_container_width=True)

    elif calc_type == "مقارنة الكتب":
        st.subheader("🏆 مقارنة الكتب وإيجاد أفضل أوديد")
        st.markdown("أدخل أوديدات الفوز للفريق الأول من كل كتاب:")
        bookie_comparison = {}
        num_comp = st.slider("عدد الكتب للمقارنة", 2, 8, 5)
        comp_cols = st.columns(min(num_comp, 4))
        bookie_list = ["Bet365", "William Hill", "Pinnacle", "Betfair", "Unibet", "888sport", "Ladbrokes", "Bwin"]

        for i in range(num_comp):
            col = comp_cols[i % 4]
            bname = bookie_list[i] if i < len(bookie_list) else f"كتاب {i+1}"
            o = col.number_input(bname, 1.01, 50.0, 2.0, 0.01, key=f"comp_{i}")
            bookie_comparison[bname] = o

        best_bookie = max(bookie_comparison, key=bookie_comparison.get)
        best_odd = bookie_comparison[best_bookie]
        avg_odd = np.mean(list(bookie_comparison.values()))

        c1, c2, c3 = st.columns(3)
        c1.metric("أفضل كتاب", best_bookie)
        c2.metric("أفضل أوديد", f"{best_odd:.3f}")
        c3.metric("الفرق عن المتوسط", f"{(best_odd - avg_odd):.3f}")

        fig_comp = px.bar(
            x=list(bookie_comparison.keys()),
            y=list(bookie_comparison.values()),
            title="مقارنة الأوديدات",
            color=list(bookie_comparison.values()),
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="fig_comp_books")

    elif calc_type == "حساب CLV":
        st.subheader("📉 Closing Line Value (CLV)")
        st.markdown("""
        **CLV** هو أهم مقياس في المراهنات الاحترافية.
        إذا كنت تراهن باستمرار بأوديدات أفضل من الإغلاق = أنت تتفوق على السوق.
        """)
        col1, col2 = st.columns(2)
        opening = col1.number_input("أوديد الافتتاح (وقت رهانك)", 1.01, 50.0, 2.20, 0.01)
        closing = col2.number_input("أوديد الإغلاق", 1.01, 50.0, 2.05, 0.01)

        processor = OddsProcessor()
        clv = processor.compute_closing_line_value(opening, closing)

        if clv > 0:
            st.success(f"✅ CLV إيجابي: +{clv:.2f}% — أنت تتفوق على السوق!", icon="⭐")
        elif clv < 0:
            st.error(f"❌ CLV سلبي: {clv:.2f}% — السوق تتفوق عليك", icon="⚠️")
        else:
            st.info("CLV = 0 — مُعادل للسوق")

        st.metric("Closing Line Value", f"{clv:+.2f}%")

# ============================ Footer ============================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.markdown("🔗 **[The Odds API](https://the-odds-api.com)**")
col_f2.markdown("📊 **منهجية:** Shin + Kelly + Value Betting")
col_f3.markdown("⚠️ **للأغراض التعليمية فقط**")
