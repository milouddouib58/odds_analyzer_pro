# data_loader.py (النسخة المصححة مع ترجمة الأعمدة)
import pandas as pd
import requests
import os

# قاموس يحتوي على روابط التحميل المباشرة لكل دوري
LEAGUE_CSV_URLS = {
    "E0.csv": "https://www.football-data.co.uk/mmz4281/2425/E0.csv", # Premier League
    "SP1.csv": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv", # La Liga
    "I1.csv": "https://www.football-data.co.uk/mmz4281/2425/I1.csv", # Serie A
    "D1.csv": "https://www.football-data.co.uk/mmz4281/2425/D1.csv", # Bundesliga
    "F1.csv": "https://www.football-data.co.uk/mmz4281/2425/F1.csv", # Ligue 1
}

def load_stats_data_from_csv(filepath: str):
    """
    يقوم بتحميل بيانات الإحصائيات من ملف CSV.
    إذا لم يكن الملف موجودًا، يقوم بتحميله تلقائيًا.
    ويقوم بإعادة تسمية الأعمدة لتكون متوافقة.
    """
    if not os.path.exists(filepath):
        print(f"⚠️ الملف '{filepath}' غير موجود. جاري محاولة تحميله تلقائيًا...")
        filename = os.path.basename(filepath)
        url = LEAGUE_CSV_URLS.get(filename)
        if not url:
            print(f"❌ لا يوجد رابط تحميل معروف للملف '{filename}'.")
            return None
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ تم تحميل '{filepath}' بنجاح.")
        except requests.exceptions.RequestException as e:
            print(f"❌ فشل تحميل الملف من الرابط: {e}")
            return None

    try:
        # --- ::: التعديل الأهم يبدأ هنا ::: ---
        
        # 1. تحميل الملف
        df = pd.read_csv(filepath, encoding='ISO-8859-1', on_bad_lines='skip')

        # 2. قاموس الترجمة لأسماء الأعمدة
        rename_map = {
            'FTHG': 'HomeGoals',
            'FTAG': 'AwayGoals',
            'FTR': 'Result',
            'HS': 'HomeShots',
            'AS': 'AwayShots',
            'HST': 'HomeShotsTarget',
            'AST': 'AwayShotsTarget',
            'xG': 'Home_xG',
            'xG.1': 'Away_xG',
            'xGA': 'Away_xG_for_Home',
            'xGA.1': 'Home_xG_for_Away'
        }
        
        # 3. إعادة تسمية الأعمدة الموجودة فقط
        df.rename(columns=lambda c: rename_map.get(c, c), inplace=True)
        
        # التأكد من وجود الأعمدة الأساسية بعد إعادة التسمية
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
        if not all(col in df.columns for col in required_cols):
            print("❌ الأعمدة الأساسية (HomeGoals, AwayGoals, etc.) غير موجودة في ملف CSV.")
            return None

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        print("✅ تم تحميل وترجمة أعمدة ملف الإحصائيات بنجاح.")
        return df
        # --- ::: نهاية التعديل ::: ---
        
    except Exception as e:
        print(f"Error loading stats CSV: {e}")
        return None
