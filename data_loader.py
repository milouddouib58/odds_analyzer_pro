# data_loader.py (النسخة النهائية مع قاموس الترجمة)
import pandas as pd
import requests
import os

# --- ::: قاموس الترجمة الذكي ::: ---
# هذا هو الجزء الأهم لحل المشكلة بشكل نهائي.
# المفتاح: هو اسم الفريق كما يأتي من The Odds API (بحروف صغيرة).
# القيمة: هو اسم الفريق كما هو مكتوب بالضبط في ملف CSV.
TEAM_NAME_MAP = {
    # La Liga (SP1.csv)
    "sevilla": "Sevilla",
    "elche cf": "Elche",
    "real betis": "Betis",
    "celta de vigo": "Celta",
    "cádiz cf": "Cadiz",
    "atlético madrid": "Atletico Madrid",
    "real sociedad": "Sociedad",
    "fc barcelona": "Barcelona",
    "real valladolid": "Valladolid",
    "deportivo alavés": "Alaves",
    
    # Premier League (E0.csv) - كمثال
    "manchester united": "Man United",
    "manchester city": "Man City",
    "wolverhampton wanderers": "Wolves",
    "nottingham forest": "Nott'm Forest",
    "brighton and hove albion": "Brighton",
    "west ham united": "West Ham",
    "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham",
}

def get_csv_name(odds_api_name: str) -> str:
    """
    يترجم اسم الفريق من مصدر الأسعار إلى الاسم الموجود في ملف CSV.
    """
    normalized_name = odds_api_name.lower().strip()
    return TEAM_NAME_MAP.get(normalized_name, odds_api_name) # يرجع الاسم الأصلي إذا لم يجد ترجمة

def load_stats_data_from_csv(filepath: str):
    """
    يقوم بتحميل بيانات الإحصائيات من ملف CSV ويقوم بإعادة تسمية الأعمدة.
    """
    if not os.path.exists(filepath):
        st.error(f"الملف '{filepath}' غير موجود. الرجاء تحميله ووضعه في المجلد.")
        return None
        
    try:
        df = pd.read_csv(filepath, encoding='ISO-8851', on_bad_lines='skip')
        
        rename_map = {
            'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'FTR': 'Result',
            'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
            'xG': 'Home_xG', 'xGA': 'Away_xG_for_Home',
            'xG.1': 'Away_xG', 'xGA.1': 'Home_xG_for_Away'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
        if not all(col in df.columns for col in required_cols):
            print(f"الأعمدة المطلوبة غير موجودة. الأعمدة المتوفرة: {df.columns.tolist()}")
            return None

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        print(f"حدث خطأ أثناء قراءة ملف CSV: {e}")
        return None
