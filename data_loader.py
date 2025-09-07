# data_loader.py
import pandas as pd

def load_stats_data_from_csv(filepath: str):
    """
    يقوم بتحميل بيانات الإحصائيات من ملف CSV الخاص بـ football-data.co.uk.
    """
    try:
        # تحديد الأعمدة المهمة فقط لتسريع العملية
        columns_to_use = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'xG', 'xGA', 'xG.1', 'xGA.1']
        df = pd.read_csv(filepath, usecols=columns_to_use, encoding='ISO-8859-1')
        df.columns = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'Result', 
                      'HomeShots', 'AwayShots', 'HomeShotsTarget', 'AwayShotsTarget',
                      'Home_xG', 'Away_xG_for_Home', 'Away_xG', 'Home_xG_for_Away']
        
        # تحويل التاريخ إلى صيغة مفهومة
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        return df
    except Exception as e:
        print(f"Error loading stats CSV: {e}")
        return None
