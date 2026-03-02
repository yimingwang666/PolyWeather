import requests
import pandas as pd
import numpy as np
import json
import os
import math
from datetime import datetime
import pytz
from scipy.interpolate import interp1d
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# === 配置项 ===
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
MODEL_PATH = "rksi_model.cbm"
OUTPUT_JSON = "data.json"

# 映射名称，用于前端展示更清晰
MODEL_NAMES = {
    'temperature_2m_ecmwf_ifs025': 'ECMWF (欧洲)',
    'temperature_2m_gfs_seamless': 'GFS (美国)',
    'temperature_2m_icon_seamless': 'ICON (德国)',
    'temperature_2m_jma_seamless': 'JMA (日本)',
    'temperature_2m_ukmo_seamless': 'UKMO (英国)',
    'temperature_2m_cma_grapes_global': 'CMA (中国)',
    'temperature_2m_archive_best_match': 'Best Match (综合)'
}

def clean_float(val):
    """清理异常浮点数，供 JSON 序列化"""
    if val is None or pd.isna(val) or math.isnan(val):
        return None
    return float(round(val, 2))

def fetch_realtime_data():
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    today_str = now.strftime('%Y%m%d')
    current_hour = now.hour
    
    # 1. 抓取 Wunderground (RKSI) 实况数据
    wu_url = f"https://api.weather.com/v1/location/RKSI:9:KR/observations/historical.json?apiKey={WU_API_KEY}&units=m&startDate={today_str}&endDate={today_str}"
    
    current_temp = max_temp_so_far = rh = wdir = wspd = pressure = np.nan
    actual_temp_24h = [None] * 24
    
    try:
        wu_resp = requests.get(wu_url, timeout=15).json()
        obs = wu_resp.get('observations', [])
        if obs:
            df_wu = pd.DataFrame(obs)
            max_temp_so_far = df_wu['temp'].max()
            
            latest = df_wu.iloc[-1]
            current_temp = latest['temp']
            rh = latest['rh']
            wdir = latest['wdir']
            wspd = latest['wspd']
            pressure = latest['pressure']
            
            # 构建今日24小时真实温度曲线
            df_wu['datetime'] = pd.to_datetime(df_wu['valid_time_gmt'], unit='s', utc=True).dt.tz_convert('Asia/Seoul')
            df_wu['hr'] = df_wu['datetime'].dt.hour
            hourly_actual = df_wu.groupby('hr')['temp'].mean().to_dict()
            for h in range(24):
                # 只有当前或之前的小时才有实际数据
                if h <= current_hour:
                    actual_temp_24h[h] = clean_float(hourly_actual.get(h, None))
    except Exception as e:
        print(f"Wunderground 失败: {e}")

    # 2. 抓取 Open-Meteo 24小时预测
    om_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.49&longitude=126.49"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation"
        "&models=ecmwf_ifs025,gfs_seamless,icon_seamless,jma_seamless,ukmo_seamless,cma_grapes_global,best_match"
        "&timezone=Asia%2FSeoul&forecast_days=1"
    )
    
    om_data_current = {}
    chart_forecasts = {}
    
    try:
        om_resp = requests.get(om_url, timeout=15).json()
        hourly = om_resp.get('hourly', {})
        
        for key, values in hourly.items():
            if key == 'time': continue
            feature_name = key.replace('_best_match', '_archive_best_match')
            om_data_current[feature_name] = values[current_hour]
            
            # 如果是温度，保存全天 24 小时用于画图
            if key in MODEL_NAMES:
                chart_forecasts[MODEL_NAMES[key]] = [clean_float(v) for v in values[:24]]
                
    except Exception as e:
        print(f"Open-Meteo 失败: {e}")

    return {
        "update_time": now.strftime('%Y-%m-%d %H:%M:%S KST'),
        "hour": current_hour,
        "month": now.month,
        "wu_realtime": {
            "temp": current_temp,
            "max_temp_so_far": max_temp_so_far,
            "rh": rh,
            "wdir": wdir,
            "wspd": wspd,
            "pressure": pressure
        },
        "om_forecast": om_data_current,
        "chart_data": {
            "hours": [f"{i:02d}:00" for i in range(24)],
            "actual_temp": actual_temp_24h,
            "forecasts": chart_forecasts
        }
    }

def run_bot():
    data = fetch_realtime_data()
    feature_dict = {}
    
    feature_dict['temp'] = data['wu_realtime']['temp']
    feature_dict['max_temp_so_far'] = data['wu_realtime']['max_temp_so_far']
    feature_dict['rh'] = data['wu_realtime']['rh']
    feature_dict['wdir'] = data['wu_realtime']['wdir']
    feature_dict['wspd'] = data['wu_realtime']['wspd']
    feature_dict['pressure'] = data['wu_realtime']['pressure']
    
    feature_dict.update(data['om_forecast'])
    df_features = pd.DataFrame([feature_dict]).fillna(method='bfill', axis=1).fillna(0)
    
    temp_cols = [c for c in df_features.columns if c.startswith('temperature_2m_')]
    df_features['forecast_temp_mean'] = df_features[temp_cols].mean(axis=1)
    df_features['forecast_temp_std'] = df_features[temp_cols].std(axis=1)
    
    current_hour = data['hour']
    current_month = data['month']
    df_features['hour'] = current_hour
    df_features['month'] = current_month
    df_features['hour_sin'] = np.sin(2 * np.pi * current_hour / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * current_hour / 24)
    df_features['month_sin'] = np.sin(2 * np.pi * current_month / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * current_month / 12)
    
    # 提取用于前端展示的【当前时刻机构预测明细】
    inst_current = []
    for raw_col, display_name in MODEL_NAMES.items():
        adj_col = raw_col.replace('_best_match', '_archive_best_match')
        val = df_features[adj_col].iloc[0] if adj_col in df_features.columns else None
        inst_current.append({"name": display_name, "temp": clean_float(val)})
    
    # --- 模型推理 ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    
    expected_cols = model.feature_names_
    for c in expected_cols:
        if c not in df_features.columns:
            df_features[c] = 0.0
    X_live = df_features[expected_cols]

    raw_quantiles = np.sort(model.predict(X_live)[0])
    median_temp = raw_quantiles[2]
    
    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    cdf_interp = interp1d(raw_quantiles, alphas, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    target_temps = [int(median_temp)-2, int(median_temp)-1, int(median_temp), int(median_temp)+1, int(median_temp)+2]
    
    probs = []
    for t in target_temps:
        prob = cdf_interp(t + 0.5) - cdf_interp(t - 0.5)
        probs.append({"temp": int(t), "prob": clean_float(prob * 100)})
    probs = sorted(probs, key=lambda x: x['prob'], reverse=True)
    
    # 输出 JSON
    output = {
        "update_time": data['update_time'],
        "hour": int(current_hour),
        "realtime": {
            "current_temp": clean_float(data['wu_realtime']['temp']),
            "max_temp": clean_float(data['wu_realtime']['max_temp_so_far']),
            "forecast_mean": clean_float(df_features['forecast_temp_mean'].iloc[0])
        },
        "institutions": inst_current,
        "chart_data": data['chart_data'],
        "model": {
            "median": clean_float(median_temp),
            "quantiles": { "p05": clean_float(raw_quantiles[0]), "p95": clean_float(raw_quantiles[4]) },
            "probabilities": probs
        }
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_bot()
    print(f"数据更新完成，已写入 {OUTPUT_JSON}")
