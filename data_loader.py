# data_loader.py (النسخة الأوتوماتيكية)
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
    """
    # التحقق إذا كان الملف موجودًا
    if not os.path.exists(filepath):
        print(f"⚠️ الملف '{filepath}' غير موجود. جاري محاولة تحميله تلقائيًا...")
        
        # استخراج اسم الملف للعثور على الرابط
        filename = os.path.basename(filepath)
        url = LEAGUE_CSV_URLS.get(filename)
        
        if not url:
            print(f"❌ لا يوجد رابط تحميل معروف للملف '{filename}'.")
            return None
        
        try:
            # تحميل الملف من الإنترنت
            response = requests.get(url)
            response.raise_for_status() # التأكد من نجاح الطلب
            
            # كتابة محتوى الملف محليًا
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ تم تحميل '{filepath}' بنجاح.")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ فشل تحميل الملف من الرابط: {e}")
            return None

    # الآن، نقرأ الملف (سواء كان موجودًا من قبل أو تم تحميله الآن)
    try:
        columns_to_use = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'xG', 'xGA', 'xG.1', 'xGA.1']
        df = pd.read_csv(filepath, usecols=lambda c: c in columns_to_use, encoding='ISO-8859-1')
        
        # إعادة تسمية الأعمدة لتكون متوافقة مع xG
        rename_map = {
            'xG': 'Home_xG',
            'xG.1': 'Away_xG',
            'xGA': 'Away_xG_for_Home', # xG conceded by home team
            'xGA.1': 'Home_xG_for_Away'  # xG conceded by away team
        }
        df.rename(columns=rename_map, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        return df
    except Exception as e:
        print(f"Error loading stats CSV: {e}")
        return None

