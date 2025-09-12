# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from collections import OrderedDict

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_curve, precision_recall_curve, accuracy_score,
    balanced_accuracy_score, f1_score, roc_auc_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)

# محاولات استيراد اختيارية لنماذج إضافية
HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass

# إعدادات الصفحة
st.set_page_config(
    page_title="المختبر الاستراتيجي للجولات (نسخة مطوّرة)",
    page_icon="🧪",
    layout="wide"
)

# ثوابت عامة
RANDOM_STATE = 42
N_SPLITS = 5
MIN_REQUIRED_VALUES = 10
EPS = 1e-6

# ============================ أدوات مساعدة ============================

def _streak_series(binary_series: pd.Series) -> pd.Series:
    groups = (binary_series != binary_series.shift()).cumsum()
    streak = binary_series.groupby(groups).cumcount() + 1
    return streak * binary_series

def _streak_from_list(values, cond):
    c = 0
    for v in reversed(values):
        if cond(v):
            c += 1
        else:
            break
    return c

def compute_optimal_thresholds(y_true, proba):
    """
    يرجع عتبة Youden وعتبة F1 مع بعض المقاييس.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    # Youden
    fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
    J = tpr - fpr
    thr_youden = float(np.clip(roc_thresholds[int(np.argmax(J))], 0.0, 1.0))

    # F1
    precision, recall, pr_thresholds = precision_recall_curve(y_true, proba)
    if pr_thresholds.size > 0:
        f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + EPS)
        thr_f1 = float(pr_thresholds[int(np.argmax(f1_scores))])
    else:
        thr_f1 = 0.5

    # مقاييس عند youden الافتراضي
    y_pred_youden = (proba >= thr_youden).astype(int)
    metrics = {
        'acc': float(accuracy_score(y_true, y_pred_youden)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred_youden)),
        'f1_at_youden': float(f1_score(y_true, y_pred_youden)),
        'roc_auc': float(roc_auc_score(y_true, proba))
    }

    return {'youden': thr_youden, 'f1': thr_f1}, metrics

# ============================ هندسة السمات ============================

def build_features_df(raw_data: list) -> pd.DataFrame:
    """
    يبني DataFrame ميزات متقدمة من السلسلة التاريخية.
    الهدف: target = 1 إذا crash >= 2.0 وإلا 0.
    """
    s = pd.Series(raw_data, name="crash")
    df = pd.DataFrame(s)

    # أعلام وسلاسل
    df['is_low'] = (df['crash'] < 2.0).astype(int)
    df['is_high'] = (df['crash'] >= 2.0).astype(int)
    df['low_streak'] = _streak_series(df['is_low'])
    df['high_streak'] = _streak_series(df['is_high'])

    # ميزات تُبنى من القيم السابقة فقط
    x = df['crash'].shift(1)
    feats = pd.DataFrame(index=df.index)

    # لاقات
    feats['lag_1'] = x
    feats['lag_2'] = df['crash'].shift(2)
    feats['lag_3'] = df['crash'].shift(3)

    # إحصاءات متحركة
    for w in [3, 5, 10]:
        feats[f'avg_last_{w}'] = x.rolling(window=w, min_periods=w).mean()
        feats[f'std_last_{w}'] = x.rolling(window=w, min_periods=w).std(ddof=1)

    feats['min_last_5'] = x.rolling(5, min_periods=5).min()
    feats['max_last_5'] = x.rolling(5, min_periods=5).max()
    feats['median_last_5'] = x.rolling(5, min_periods=5).median()
    feats['range_last_5'] = feats['max_last_5'] - feats['min_last_5']

    # EMA
    feats['ema_3'] = x.ewm(span=3, adjust=False).mean()
    feats['ema_5'] = x.ewm(span=5, adjust=False).mean()
    feats['ema_10'] = x.ewm(span=10, adjust=False).mean()

    # سلاسل ونسب
    feats['low_streak'] = df['low_streak'].shift(1)
    feats['high_streak'] = df['high_streak'].shift(1)
    feats['low_ratio_last_10'] = df['is_low'].shift(1).rolling(10, min_periods=10).mean()

    # مشتقات ونِسب تذبذب
    feats['delta_prev_vs_avg5'] = feats['lag_1'] - feats['avg_last_5']
    feats['zscore_prev_5'] = feats['delta_prev_vs_avg5'] / (feats['std_last_5'] + EPS)
    feats['vol_ratio_3_5'] = feats['std_last_3'] / (feats['std_last_5'] + EPS)
    feats['vol_ratio_5_10'] = feats['std_last_5'] / (feats['std_last_10'] + EPS)

    # الهدف
    feats['target'] = (df['crash'] >= 2.0).astype(int)

    feats.dropna(inplace=True)
    feats.reset_index(drop=True, inplace=True)
    return feats

def build_feature_row_from_values(last_values: list) -> dict:
    """
    يبني صف ميزات من تاريخ قيم (≥ 10).
    """
    if len(last_values) < MIN_REQUIRED_VALUES:
        raise ValueError(f"أدخل {MIN_REQUIRED_VALUES} قيماً على الأقل.")

    v = list(map(float, last_values))
    last3, last5, last10 = v[-3:], v[-5:], v[-10:]

    lag_1 = float(v[-1])
    lag_2 = float(v[-2])
    lag_3 = float(v[-3])

    avg_last_3 = float(np.mean(last3))
    std_last_3 = float(np.std(last3, ddof=1))

    avg_last_5 = float(np.mean(last5))
    std_last_5 = float(np.std(last5, ddof=1))

    avg_last_10 = float(np.mean(last10))
    std_last_10 = float(np.std(last10, ddof=1))

    min_last_5 = float(np.min(last5))
    max_last_5 = float(np.max(last5))
    median_last_5 = float(np.median(last5))
    range_last_5 = max_last_5 - min_last_5

    ema_3 = float(pd.Series(v).ewm(span=3, adjust=False).mean().iloc[-1])
    ema_5 = float(pd.Series(v).ewm(span=5, adjust=False).mean().iloc[-1])
    ema_10 = float(pd.Series(v).ewm(span=10, adjust=False).mean().iloc[-1])

    low_streak = _streak_from_list(v, lambda x: x < 2.0)
    high_streak = _streak_from_list(v, lambda x: x >= 2.0)
    low_ratio_last_10 = float(np.mean(np.array(last10) < 2.0))

    delta_prev_vs_avg5 = lag_1 - avg_last_5
    zscore_prev_5 = delta_prev_vs_avg5 / (std_last_5 + EPS)
    vol_ratio_3_5 = std_last_3 / (std_last_5 + EPS)
    vol_ratio_5_10 = std_last_5 / (std_last_10 + EPS)

    return {
        'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3,
        'avg_last_3': avg_last_3, 'std_last_3': std_last_3,
        'avg_last_5': avg_last_5, 'std_last_5': std_last_5,
        'avg_last_10': avg_last_10, 'std_last_10': std_last_10,
        'min_last_5': min_last_5, 'max_last_5': max_last_5,
        'median_last_5': median_last_5, 'range_last_5': range_last_5,
        'ema_3': ema_3, 'ema_5': ema_5, 'ema_10': ema_10,
        'low_streak': float(low_streak), 'high_streak': float(high_streak),
        'low_ratio_last_10': low_ratio_last_10,
        'delta_prev_vs_avg5': float(delta_prev_vs_avg5),
        'zscore_prev_5': float(zscore_prev_5),
        'vol_ratio_3_5': float(vol_ratio_3_5),
        'vol_ratio_5_10': float(vol_ratio_5_10)
    }

# ============================ توليد OOF وتجميع ============================

def get_base_models():
    models = OrderedDict()

    models['RF'] = RandomForestClassifier(
        n_estimators=800, max_features='sqrt',
        min_samples_split=10, min_samples_leaf=4,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )

    models['ET'] = ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt',
        min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )

    models['GB'] = GradientBoostingClassifier(
        learning_rate=0.05, n_estimators=400, max_depth=3, random_state=RANDOM_STATE
    )

    models['HGB'] = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=None, max_iter=400, random_state=RANDOM_STATE
    )

    models['LR'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE
        ))
    ])

    models['SVC'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_STATE
        ))
    ])

    if HAS_XGB:
        models['XGB'] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss',
            tree_method='hist'
        )
    if HAS_LGBM:
        models['LGBM'] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
        )

    return models

def generate_oof_and_fit(models_dict, X, y, n_splits=5):
    """
    يرجع:
    - oof_df: إطار يحتوي احتمالات OOF لكل نموذج
    - fitted_models: النماذج المدربة على كامل البيانات
    """
    cv = TimeSeriesSplit(n_splits=n_splits)
    oof_df = pd.DataFrame(index=X.index, columns=list(models_dict.keys()), dtype=float)
    fitted_models = {}

    for name, model in models_dict.items():
        oof_vals = np.full(len(X), np.nan, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            mdl = clone(model)
            mdl.fit(X.iloc[tr_idx], y[tr_idx])
            oof_vals[va_idx] = mdl.predict_proba(X.iloc[va_idx])[:, 1]
        oof_df[name] = oof_vals

        # fit على كامل البيانات للاستخدام في التنبؤ التفاعلي
        mdl_full = clone(model)
        mdl_full.fit(X, y)
        fitted_models[name] = mdl_full

    valid_mask = ~oof_df.isna().any(axis=1)
    return oof_df[valid_mask].reset_index(drop=True), fitted_models, valid_mask

# ============================ تدريب شامل مع تجميع ============================

@st.cache_resource
def train_and_prepare():
    # 1) البيانات الأصلية (كما وفّرتها)
    raw_data = [
        8.72, 6.75, 1.86, 2.18, 1.25, 2.28, 1.24, 1.2, 1.54, 24.46, 4.16, 1.49, 1.09, 1.47, 1.54, 1.53, 2.1, 32.04, 11, 1.17, 1.7, 2.61, 1.26, 22.23, 1.77, 1.93, 3.35, 7.01, 1.83, 9.39, 3.31, 2.04, 1.3, 6.65, 1.16, 3.39, 1.95, 10.85, 1.65, 1.22, 1.6, 4.67, 1.85, 2.72, 1, 3.02, 1.35, 1.3, 1.37, 17.54, 1.18, 1, 14.4, 1.11, 6.15, 2.39, 2.22, 1.42, 1.23, 2.42, 1.07, 1.24, 2.55, 7.26, 1.69, 5.1, 2.59, 5.51, 2.31, 2.12, 1.97, 1.5, 3.01, 2.29, 1.36, 4.95, 5.09, 8.5, 1.77, 5.52, 3.93, 1.5, 2.28, 2.49, 18.25, 1.68, 1.42, 2.12, 4.17, 1.04, 2.35, 1, 1.01, 5.46, 1.13, 2.84, 3.39, 2.79, 1.59, 1.53, 4.34, 2.96, 1.06, 1.72, 2.16, 2.2, 3.61, 2.34, 4.49, 1.72, 1.78, 9.27, 8.49, 2.86, 1.66, 4.63, 9.25, 1.35, 1, 1.64, 1.86, 2.81, 2.44, 1.74, 1.1, 1.29, 1.45, 8.92, 1.24, 6.39, 1.16, 1.19, 2.4, 4.64, 3.17, 24.21, 1.17, 1.42, 2.13, 1.12, 3.78, 1.12, 1.52, 22.81, 1.31, 1.9, 1.38, 1.47, 2.86, 1.79, 1.49, 1.38, 1.84, 1.06, 3.3, 5.97, 1, 2.92, 1.64, 5.32, 3.26, 1.78, 2.24, 3.16, 1.6, 1.08, 1.55, 1.07, 1.02, 1.23, 1.08, 5.22, 3.32, 24.86, 3.37, 5.16, 1.69, 2.31, 1.07, 1.1, 1.01, 1.36, 1.38, 1.54, 5.34, 2.68, 5.78, 3.63, 1.89, 8.41, 4.06, 1.44, 1.5, 3.17, 1.02, 1.8, 1.9, 1.86, 1.85, 1.73, 3.86, 3.11, 2.44, 1.15, 2.03, 1.05, 3.05, 1.88, 10.13, 2.29, 1.41, 1, 5.46, 1.26, 23.33, 1.96, 1.03, 4.54, 1.37, 3.5, 1.13, 1.16, 1.43, 1.13, 1.05, 33.27, 9.96, 1.79, 2.07, 18.51, 5.75, 1.15, 1.08, 5.92, 1.38, 1.61, 12.99, 24.72, 4.86, 1.11, 2.86, 1.54, 3.71, 4, 7.57, 2.03, 2.18, 5.52, 13.37, 3.73, 2.41, 1.79, 5.57, 4.36, 12.33, 1.61, 3.28, 2.89, 1.47, 1.08, 26.89, 1.53, 2.94, 5.29, 1.23, 1.57, 1.12, 5.69, 3.29, 2.72, 1.18, 5.03, 1.1, 1.32, 1.18, 1.07, 1.27, 4.6, 11.68, 1.74, 3.94, 3.63, 1.05, 1.61, 1.62, 2.41, 6.9, 2.02, 1.01, 3.22, 17.21, 1.95, 8.8, 1.44, 2.76, 3.1, 2.84, 1.35, 1.84, 1.6, 10.72, 1.17, 3.47, 1.45, 1.29, 1.46, 2.23, 12.3, 3.27, 1.23, 1.02, 1.66, 3.79, 2.06, 4.55, 7.95, 8.55, 4.08, 2.02, 1.21, 1.19, 1.53, 4.9, 1.84, 10.51, 1.01, 1.34, 1.5, 1.4, 1.42, 4.18, 7.99, 1.23, 1.67, 3.16, 1.64, 25.06, 4.52, 1.5, 3.23, 1.09, 1.45, 2.77, 7.42, 7.48, 1.89, 2.11, 4.1, 1.26, 2.29, 10.12, 1.35, 13.21, 2.36, 22.35, 1.76, 2.22, 1.04, 1.18, 3.69, 1.47, 10.2, 1.47, 1.68, 2.45, 1.03, 2.04, 1.47, 1.18, 1.72, 1, 3.25, 1.1, 8.74, 1.01, 1.54, 1.34, 5.22, 5.31, 4.47, 2.78, 21.37, 3.38, 1.63, 2.21, 2.35, 2.14, 1.46, 1.25, 1.67, 1.08, 3.94, 1.66, 31.1, 1.73, 2.18, 2.06, 1.08, 1.11, 1, 1.07, 1.31, 1.55, 1.98, 1.75, 1.23, 1.32, 2.56, 3.21, 1.81, 2.09, 1.34, 3.42, 1.29, 1.36, 1.76, 1.61, 4.52, 1.08, 1.97, 3.75, 1.8, 6.36, 1.14, 1.72, 2.39, 1.28, 4.22, 2.12, 1.28, 1.38, 1.42, 28.26, 2.15, 1.31, 1.65, 2.43, 2.76, 1.54, 1.61, 11.91, 2.93, 8.1, 2.04, 1.84, 1.26, 3.69, 3.97, 3.01, 3.16, 1.3, 7.9, 1.72, 5.57, 2.42, 1.74, 2.06, 2.86, 1.56, 1.4, 2.35, 2.82, 4.03, 1.28, 2.21, 1.1, 2.06, 1.14, 1.58, 27.78, 2.04, 1.52, 1.22, 1.4, 1.29, 1.16, 11.72, 1.33, 1.3, 4.34, 1.02, 1.63, 1.9, 9, 1.42, 3.13, 3.8, 1.02, 1.25, 2.45, 1.74, 1.06, 1.38, 3.46, 1.08, 1, 1.02, 1.84, 1, 1.77, 3.07, 5.26, 1.73, 1.07, 3.75, 2.32, 1.6, 1.22, 1.72, 2.01, 1.11, 2.03, 1.17, 1.98, 2.18, 34.49, 1.2, 10.3, 3.4, 2.58, 2.2, 3.16, 29.22, 4.26, 3.18, 3.29, 1.09, 2.3, 1.25, 3.05, 2.99, 2.16, 3.02, 2.21, 1.59, 5.74, 1.02, 1.12, 1.21, 2.25, 4.38, 1.05, 1.05, 1.9, 23.03, 4.93, 1.03, 16.7, 4.08, 1.68, 2.4, 2.89, 2.85, 2.75, 20.29, 3.57, 9.68, 1.46, 5.73, 4.84, 1.15, 1.92, 3.71, 3.41, 22.67, 15.65, 1.86, 3.41, 1.89, 1.01, 3.02, 13.81, 1.55, 1.16, 6.35, 5.6, 2.55, 16.8, 5.48, 1.49, 2.07, 1.05, 1.49, 6.29, 1.32, 23.22, 1.07, 1.65, 20.07, 1.14, 1.1, 18.38, 4.34, 3.8, 6.17, 2.27, 1.69, 1.07, 3.74, 1.6, 1.02, 1.45, 1.86, 5.13, 1.57, 6.93, 15.82, 1, 1.16, 4.14, 1.08, 2.35, 2.15, 13.52, 10.87, 9.85, 1.97, 1, 3.46, 1.31, 3.28, 2.74, 1.98, 2.22, 1, 9.95, 1.41, 1.43, 2.13, 4.6, 2.68, 4.13, 1.61, 1.46, 1.23, 9.57, 1.14, 1.17, 14.27, 4.01, 5.55, 1.95, 2.48, 1.78, 2.21, 1.65, 1.08, 2.63, 8.53, 2.2, 1.33, 21.72, 1.3, 1.43, 6.37, 1.09, 3.94, 1.88, 3.38, 1.66, 1.41, 22.99, 1.55, 7.5, 25.48, 2.21, 3.62, 1.68, 9.92, 3.4, 2.66, 1.03, 4.63, 1.89, 1.77, 1.9, 1.01, 1.81, 32.39, 2.1, 1.23, 6.26, 9.06, 1.17, 2.41, 2.52, 1.63, 5.61, 1, 2.63, 1.88, 1.5, 23.8, 5.65, 1.05, 1.07, 2.05, 1.7, 2.4, 18.27, 3.68, 13.17, 4.99, 20.81, 1.51, 6.33, 9.85, 10.15, 17.05, 27.6, 4.65, 3.18, 2.54, 3.92, 4.74, 1.81, 1.91, 4.42, 1.57, 2.17, 1.25, 1.03, 1.15, 1.19, 13.97, 2.39, 1.34, 2.52, 1.47, 2.91, 2.31, 1.29, 1.61, 4.13, 1.83, 2.96, 1.08, 1.28, 13.53, 1.15, 1.51, 1.31, 3.45, 9.32, 5.42, 3.27, 2.56, 2.07, 1.83, 14.1, 15.36, 1.93, 1.47, 16.96, 1.61, 2.38, 2.66, 1.28, 1.46, 3.09, 6.73, 1.12, 1.85, 3.21, 1.15, 3.71, 1.64, 4.88, 11.09, 3.82, 2.49, 21.23, 2.01, 2.47, 2.47, 2.19, 2.14, 1, 2.09, 1.03, 5.22, 1.65, 1.13, 14.43, 1.68, 1.86, 1.21, 1.14, 1.47, 1.26, 3.44, 23.9, 2.53, 2.72, 1, 1.13, 3.34, 1.43, 1, 2.48, 2.01, 2.22, 6.43, 1.81, 2.12, 1.3, 4.02, 1.79, 3.9, 1.3, 5.04, 1.77, 6.67, 2.21, 1.58, 5.38, 2.79, 6.12, 2.95, 1.14, 1.19, 1.19, 10.23, 17.96, 10.1, 2.4, 9.29, 1.28, 4.07, 1.64, 2.1, 2.67, 1.08, 16.82, 2.83, 24.42, 1.01, 3.24, 5.05, 3.24, 1.56, 2.32, 1.23, 1.72, 3.39, 1.96, 1.18, 3.21, 23.95, 9.46, 23.12, 1.45, 3.22, 5, 2.04, 2.73, 6.28, 1.21, 14.3, 1.48, 3.3, 3.73, 4.09, 2.88, 8.83, 1.15, 4.58, 4.23, 2.34, 2, 11.38, 1.81, 1.03, 1.76, 2.41, 2.5, 5.82, 2.18, 10.19, 2.08, 18.19, 4.22, 7.78, 1.96, 1.43, 1.08, 2.38, 1.37, 1.21, 4.48, 1.64, 1.62, 21.24, 1.22, 7.99, 1.13, 1.29, 2.36, 3.94, 1.08, 1.41, 1.97, 1.41, 1.95, 1.28, 4.56, 3.35, 1.37, 1.18, 1.03, 3.67, 1.43, 1.8, 2.48, 11.95, 1.5, 3.52, 2.03, 1, 1.1, 10.13, 1.44, 14.19, 2.1, 8.46, 1.06, 1.66, 1.2, 7.22, 1.75, 1.78, 3.76, 2.21, 1, 25.19, 5.96, 5.42, 2.67, 1.37, 1.39, 15.95, 2.8, 1.76, 1.7, 2.81, 8.87, 1.48, 1.03, 1.14, 1.05, 10.29, 1.71, 23.98, 2.34, 1.97, 1.33, 24.02, 2.01, 13.74, 2.5, 1.33, 1.02, 1.76, 1.37, 8.97, 1.27, 1.38, 4.47, 1.38, 3.02, 17, 13.35, 1.07, 1.38, 5.74, 6.68, 24.72, 1.47, 1.25, 4.51, 4.47, 1.99, 1.15, 4.03, 1.17, 3.42, 6.46, 1.31, 1.46, 6.67, 3.79, 1.56, 3.98, 1.62, 2.13, 1.07, 4.88, 1.62, 1.5, 6.11, 1.31, 1.85, 1.93, 1.09, 1.49, 1.41, 1.24, 1.05, 6.99, 1.33, 1.73, 10.76, 21.77, 1.18, 1.06, 5.36, 1.45, 1.16, 6.43, 2.1, 4.15, 1.14, 2.21, 33.48, 2.88, 1, 4.7, 1.27, 5.75, 4.97, 1.11, 3.51, 21.47, 1.21, 1.98, 1.11, 1.46, 1.77, 1.22, 2.65, 1.66, 5.29, 1.58, 2.03, 5.86, 1.1, 1.68, 1.35, 1.72, 1.15, 2.69, 2.81, 3.46, 1.58, 1.07, 7.18, 2.35, 6.05, 1.24, 5.69, 5.46, 1, 3.04, 4.76, 1.56, 1.41, 2.43, 7.97, 1.22, 1.94, 1.51, 21.71, 3.03, 1.43, 5.07, 1.87, 1.12, 1, 1.32, 1, 1.08, 1.1, 1.04, 1, 1.09, 1.97, 2.97, 1.21, 1.61, 5.94, 2.55, 4.48, 1.14, 2.73, 1.34, 1.33, 1.29, 1.25, 5.44, 1.77, 2.18, 2.52, 1.28, 22.25, 1.04, 3.57, 6.53, 1.34, 5.75, 1.61, 3.89, 1.07, 2.13, 5.05, 1.53, 3.53, 8.31, 2.15, 1.39, 1.23, 1.68, 17.14, 1.23, 2.38, 1, 2.02, 19.48, 1.22, 1.42, 6.26, 16.11, 2.05, 3.51, 3.53, 1.83, 6.86, 1.24, 27.78, 2.33, 3.43, 2.92, 1.26, 15.11, 24.58, 1.12, 2.46, 5.61, 9.79, 2.33, 1.34, 7.86, 1.1, 2.61, 2.34, 4.5, 1.79, 1.75, 18, 8.66, 1.92, 11.5, 1.35, 2.53, 1.79, 1.14, 1.58, 1.84, 1.35, 6.44, 4.49, 3.02, 3.16, 1.12, 1.42, 9.14, 1.26, 1.19, 2.47, 1.2, 3.88, 1.03, 1.85, 1.07, 1.03, 1.13, 4.87, 1.03, 1.8, 1.29, 6.11, 1.73, 30.16, 2.99, 2.34, 1.56, 4.33, 1.23, 7.39, 1.57, 3.16, 2.73, 1.46, 1.01, 8.24, 1.61, 2.28, 1.91, 1.49, 5.12, 3.53, 20.05, 3.26, 2.25, 6.61, 1.35, 4.32, 1, 2.13, 1.83, 1.26, 2.27, 1.21, 1.64, 1.77, 1.06, 1.05, 1.98, 3.1, 3.74, 22.09, 2.17, 2.97, 1.26, 1.83, 4.44, 1.08, 2.22, 1.24, 1.7, 20.14, 16.56, 1.72, 1.37, 1.06, 1.65, 2.42, 3.84, 1, 1.56, 1.93, 1.03, 1.47, 1.76, 12.64, 1.12, 1.32, 1.89, 1.64, 1.2, 3.15, 1.88, 1.12, 1.01, 1.45, 1.71, 1.65, 1.65, 5.16, 1.48, 1.73
    ]

    # 2) تجهيز الميزات
    feats = build_features_df(raw_data)
    X = feats.drop('target', axis=1)
    y = feats['target'].values

    # 3) النماذج الأساسية + OOF + تدريب كامل
    base_models = get_base_models()
    oof_df, fitted_models, valid_mask = generate_oof_and_fit(base_models, X, y, n_splits=N_SPLITS)
    y_oof = y[valid_mask.values] if hasattr(valid_mask, 'values') else y[valid_mask]

    # 4) المتعلم الفوقي (Stacking) على احتمالات OOF
    meta = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=5000, random_state=RANDOM_STATE)
    meta.fit(oof_df.values, y_oof)
    oof_ensemble_proba = meta.predict_proba(oof_df.values)[:, 1]

    # 5) اختيار العتبة المثلى
    thresholds, oof_metrics = compute_optimal_thresholds(y_oof, oof_ensemble_proba)

    # 6) أهمية السمات من RF للإطلاع
    rf_model = fitted_models['RF']
    try:
        importances = rf_model.feature_importances_
        feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    except Exception:
        feat_importance = pd.Series(dtype=float)

    # 7) إطار OOF للاختبار الرجعي الأمين
    backtest_df = feats.iloc[-len(y_oof):].copy()
    backtest_df.reset_index(drop=True, inplace=True)
    backtest_df['oof_proba_ensemble'] = oof_ensemble_proba
    backtest_df['actual'] = y_oof

    bundle = {
        'feature_columns': list(X.columns),
        'base_model_names': list(base_models.keys()),
        'fitted_models': fitted_models,
        'meta_learner': meta,
        'thresholds': thresholds,
        'oof_metrics': oof_metrics,
        'backtest_df': backtest_df,
        'feat_importance_rf': feat_importance
    }
    return bundle

# ============================ واجهة المستخدم ============================

st.title("🧪 المختبر الاستراتيجي للجولات — نسخة مطوّرة بالنماذج المتعددة والتجميع")
st.caption("تم تعزيز النموذج بميزات متقدمة وتجميع نماذج (Stacking) وتحسين للعتبة. لا يضمن ذلك ربحًا مستقبليًا.")

with st.spinner("⏳ جاري تجهيز الميزات وتدريب النماذج وتجميعها..."):
    trained = train_and_prepare()
st.success("✅ تم تجهيز النماذج والتجميع والاختبار الرجعي!", icon="🎉")

tab_predict, tab_backtest, tab_models = st.tabs(["🚀 أداة التنبؤ", "📈 الاختبار الرجعي", "🧠 تفاصيل النماذج"])

# ------------------------- تبويب التنبؤ -------------------------
with tab_predict:
    st.header("التنبؤ التفاعلي")
    colp1, colp2, colp3 = st.columns(3)
    thr_choice = colp1.selectbox("طريقة اختيار العتبة", ["youden", "f1"], index=0)
    conf_margin = colp2.slider("مرشح الثقة (يزيد دقة الرهانات ويقلل عددها)", 0.0, 0.4, 0.10, 0.01)
    min_vals = colp3.number_input("عدد القيم الدنيا المطلوبة", min_value=10, max_value=50, value=MIN_REQUIRED_VALUES, step=1)

    st.write("أدخل على الأقل القيم الأخيرة (سطر لكل قيمة أو مفصولة بمسافة):")
    user_input = st.text_area("القيم:", height=180, placeholder="1.23\n4.56\n...")

    if st.button("🚀 تنبأ الآن", type="primary"):
        try:
            vals = [float(x) for x in user_input.replace('\n', ' ').split()]
            if len(vals) < min_vals:
                st.error(f"أدخل {min_vals} قيماً على الأقل.", icon="🚨")
            else:
                # بناء صف ميزات
                feats_dict = build_feature_row_from_values(vals)
                X_row = pd.DataFrame([feats_dict])[trained['feature_columns']]

                # احتمالات النماذج الأساسية
                base_probs = {}
                for name in trained['base_model_names']:
                    mdl = trained['fitted_models'][name]
                    base_probs[name] = float(mdl.predict_proba(X_row)[0][1])

                # احتمال التجميع
                base_vec = np.array([base_probs[name] for name in trained['base_model_names']]).reshape(1, -1)
                proba_ens = float(trained['meta_learner'].predict_proba(base_vec)[0][1])

                thr = trained['thresholds'][thr_choice]
                pred = int(proba_ens >= thr)
                place_bet = (proba_ens >= thr + conf_margin)

                # عرض النتيجة
                if pred == 1:
                    st.success("التوقع: مرتفع (>= 2.0x) 🔼", icon="🔼")
                else:
                    st.warning("التوقع: منخفض (< 2.0x) 🔽", icon="🔽")

                c1, c2, c3 = st.columns(3)
                c1.metric("احتمال الارتفاع (التجميع)", f"{proba_ens*100:.2f}%")
                c2.metric("العتبة المختارة", f"{thr:.3f}")
                c3.metric("قرار الرهان (مرشح الثقة)", "نعم ✅" if place_bet and pred==1 else "لا/انتظار ⏳")

                # مخطط احتمالات النماذج الأساسية
                base_plot_df = pd.DataFrame({
                    'model': list(base_probs.keys()),
                    'proba_high': list(base_probs.values())
                }).sort_values('proba_high', ascending=False)

                fig = px.bar(base_plot_df, x='model', y='proba_high',
                             title="احتمالات الارتفاع حسب النماذج الأساسية",
                             range_y=[0,1])
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ خطأ في الإدخال أو التنبؤ: {e}", icon="🚨")

# ------------------------- تبويب الاختبار الرجعي -------------------------
with tab_backtest:
    st.header("الاختبار الرجعي (خارج الطيّات OOF)")
    st.write("يُستخدم هنا تنبؤ OOF للتجميع لتقييم الأداء بشكل أمين دون تفاؤل.")

    thr_choice_bt = st.selectbox("اختر العتبة", ["youden", "f1"], index=0, key="thr_bt")
    conf_margin_bt = st.slider("مرشح الثقة للرهانات", 0.0, 0.4, 0.10, 0.01, key="margin_bt")

    df_bt = trained['backtest_df'].copy()
    thr_bt = trained['thresholds'][thr_choice_bt]

    df_bt['pred_all'] = (df_bt['oof_proba_ensemble'] >= thr_bt).astype(int)
    df_bt['pred_filtered'] = (df_bt['oof_proba_ensemble'] >= (thr_bt + conf_margin_bt)).astype(int)

    # أرباح استراتيجية بسيطة
    # رهان عند توقع مرتفع: ربح 0.95 عند صحة، خسارة 1 عند خطأ
    df_bt['profit_all'] = np.where(
        (df_bt['pred_all'] == 1) & (df_bt['actual'] == 1), 0.95,
        np.where(df_bt['pred_all'] == 1, -1.0, 0.0)
    )
    df_bt['profit_filtered'] = np.where(
        (df_bt['pred_filtered'] == 1) & (df_bt['actual'] == 1), 0.95,
        np.where(df_bt['pred_filtered'] == 1, -1.0, 0.0)
    )

    df_bt['cum_profit_all'] = df_bt['profit_all'].cumsum()
    df_bt['cum_profit_filtered'] = df_bt['profit_filtered'].cumsum()

    # دقة الرهانات فقط
    bets_all = int(df_bt['pred_all'].sum())
    correct_all = int(((df_bt['pred_all'] == 1) & (df_bt['actual'] == 1)).sum())
    acc_bets_all = (correct_all / bets_all)*100 if bets_all > 0 else 0.0

    bets_f = int(df_bt['pred_filtered'].sum())
    correct_f = int(((df_bt['pred_filtered'] == 1) & (df_bt['actual'] == 1)).sum())
    acc_bets_f = (correct_f / bets_f)*100 if bets_f > 0 else 0.0

    # رسم الربح التراكمي
    plot_df = df_bt[['cum_profit_all', 'cum_profit_filtered']].copy()
    plot_df.columns = ['الربح التراكمي (كل الرهانات)', 'الربح التراكمي (مع مرشح الثقة)']
    fig2 = px.line(plot_df, title="الأداء التراكمي للاستراتيجية عبر الزمن")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

    cba1, cba2, cba3, cba4 = st.columns(4)
    cba1.metric("عدد الرهانات (كلها)", f"{bets_all}")
    cba2.metric("دقة الرهانات (كلها)", f"{acc_bets_all:.2f}%")
    cba3.metric("عدد الرهانات (بمرشح)", f"{bets_f}")
    cba4.metric("دقة الرهانات (بمرشح)", f"{acc_bets_f:.2f}%")

    # مقاييس OOF عامة
    roc_auc = trained['oof_metrics']['roc_auc']
    acc_youden = trained['oof_metrics']['acc'] * 100
    bal_acc_youden = trained['oof_metrics']['balanced_acc'] * 100
    f1_youden = trained['oof_metrics']['f1_at_youden']
    st.subheader("مقاييس OOF العامة")
    st.write(f"- ROC-AUC: {roc_auc:.3f}")
    st.write(f"- Accuracy عند عتبة Youden: {acc_youden:.2f}% | Balanced Acc: {bal_acc_youden:.2f}% | F1: {f1_youden:.3f}")

# ------------------------- تبويب تفاصيل النماذج -------------------------
with tab_models:
    st.header("تفاصيل النماذج والميزات")
    st.write("هذا التجميع يعتمد على عدة نماذج أساسية + متعلم فوقي Logistic Regression على احتمالاتها.")

    st.subheader("قائمة النماذج الأساسية")
    st.write(", ".join(trained['base_model_names']))

    if trained['feat_importance_rf'].shape[0] > 0:
        st.subheader("أكثر 15 سمة تأثيرًا (من RF)")
        st.dataframe(trained['feat_importance_rf'].head(15).to_frame("الأهمية"))
    else:
        st.info("لا تتوفر أهمية سمات (قد لا يدعم النموذج ذلك).")

st.markdown("---")
st.info("إخلاء مسؤولية: هذه أداة تعليمية/بحثية. حتى مع نماذج وتجميعات متقدمة، قد تكون النتائج عشوائية ولا تضمن ربحًا مستقبليًا. استخدام مرشح الثقة يزيد دقة الرهانات لكنه يقلل عددها.", icon="ℹ️")
