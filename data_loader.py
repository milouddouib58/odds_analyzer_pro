# data_loader.py (النسخة النهائية مع قارئ ذكي)
import pandas as pd
import requests
import os
import streamlit as st # استيراد streamlit لعرض رسائل الخطأ

# --- قاموس الترجمة الذكي (يمكنك توسيعه حسب الحاجة) ---
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
    
    # Premier League (E0.csv)
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
    return TEAM_NAME_MAP.get(normalized_name, odds_api_name)

def load_stats_data_from_csv(filepath: str):
    """
    يقوم بتحميل بيانات الإحصائيات من ملف CSV، مع معالجة أخطاء الترميز.
    """
    if not os.path.exists(filepath):
        st.error(f"الملف '{filepath}' غير موجود. الرجاء تحميله ووضعه في المجلد.")
        return None
        
    try:
        # --- ::: التعديل الأهم: محاولة القراءة بترميزات مختلفة ::: ---
        try:
            # المحاولة الأولى بالترميز القياسي
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            # المحاولة الثانية بترميز بديل وأكثر تساهلاً
            print("UTF-8 failed, trying latin1 encoding...")
            df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')

        # --- بقية الكود يبقى كما هو ---
        rename_map = {
            'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'FTR': 'Result',
            'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
            'xG': 'Home_xG', 'xGA': 'Away_xG_for_Home',
            'xG.1': 'Away_xG', 'xGA.1': 'Home_xG_for_Away'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
        if not all(col in df.columns for col in required_cols):
            st.error(f"الأعمدة الأساسية (HomeGoals, AwayGoals, etc.) غير موجودة في ملف '{filepath}'.")
            return None

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع أثناء معالجة ملف CSV: {e}")
        return None
