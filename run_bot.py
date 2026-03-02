import requests
import pandas as pd
import numpy as np
import json
import os
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

def fetch_realtime_data():
    """抓取当前最新的 Wunderground 实况 和 Open-Meteo 当前小时的预报"""
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    today_str = now.strftime('%Y%m%d')
    current_hour = now.hour
    
    # 1. 抓取 Wunderground (RKSI) 实时数据
    wu_url = f"https://api.weather.com/v1/location/RKSI:9:KR/observations/historical.json?apiKey={WU_API_KEY}&units=m&startDate={today_str}&endDate={today_str}"
    
    current_temp = max_temp_so_far = rh = wdir = wspd = pressure = np.nan
    try:
        wu_resp = requests.get(wu_url, timeout=15).json()
        obs = wu_resp.get('observations', [])
        if obs:
            df_wu = pd.DataFrame(obs)
            max_temp_so_far = df_wu['temp'].max() # 今天截至目前的最高温
            
            # 取最新的一条记录
            latest = df_wu.iloc[-1]
            current_temp = latest['temp']
            rh = latest['rh']
            wdir = latest['wdir']
            wspd = latest['wspd']
            pressure = latest['pressure']
    except Exception as e:
        print(f"Wunderground 获取失败: {e}")

    # 2. 抓取 Open-Meteo 实时预报 (对齐你训练用的 7大模型 × 9大物理量)
    om_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.49&longitude=126.49"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation"
        "&models=ecmwf_ifs025,gfs_seamless,icon_seamless,jma_seamless,ukmo_seamless,cma_grapes_global,best_match"
        "&timezone=Asia%2FSeoul&forecast_days=1"
    )
    
    om_data_current = {}
    try:
        om_resp = requests.get(om_url, timeout=15).json()
        hourly = om_resp.get('hourly', {})
        
        # 提取当前小时的数据
        for key, values in hourly.items():
            if key == 'time': continue
            # 严格对齐训练集列名：将 best_match 替换为 archive_best_match
            feature_name = key.replace('_best_match', '_archive_best_match')
            om_data_current[feature_name] = values[current_hour]
            
    except Exception as e:
        print(f"Open-Meteo 获取失败: {e}")

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
        "om_forecast": om_data_current
    }

def run_bot():
    data = fetch_realtime_data()
    feature_dict = {}
    
    # 填充基础物理实况
    feature_dict['temp'] = data['wu_realtime']['temp']
    feature_dict['max_temp_so_far'] = data['wu_realtime']['max_temp_so_far']
    feature_dict['rh'] = data['wu_realtime']['rh']
    feature_dict['wdir'] = data['wu_realtime']['wdir']
    feature_dict['wspd'] = data['wu_realtime']['wspd']
    feature_dict['pressure'] = data['wu_realtime']['pressure']
    
    # 填充 Open-Meteo 预测矩阵
    feature_dict.update(data['om_forecast'])
    
    # 填补可能出现的缺失值
    df_features = pd.DataFrame([feature_dict]).fillna(method='bfill', axis=1).fillna(0)
    
    # 完美复刻你的特有特征：计算均值和标准差
    temp_cols = [c for c in df_features.columns if c.startswith('temperature_2m_')]
    df_features['forecast_temp_mean'] = df_features[temp_cols].mean(axis=1)
    df_features['forecast_temp_std'] = df_features[temp_cols].std(axis=1)
    
    # 复刻循环时间编码
    current_hour = data['hour']
    current_month = data['month']
    df_features['hour'] = current_hour
    df_features['month'] = current_month
    df_features['hour_sin'] = np.sin(2 * np.pi * current_hour / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * current_hour / 24)
    df_features['month_sin'] = np.sin(2 * np.pi * current_month / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * current_month / 12)
    
    # --- 模型加载与推理 ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    
    # 严格对齐你训练时的底层特征顺序
    expected_cols = model.feature_names_
    for c in expected_cols:
        if c not in df_features.columns:
            df_features[c] = 0.0
    X_live = df_features[expected_cols]

    # 直接使用原生分位点，不做任何主观平滑干预
    raw_quantiles = np.sort(model.predict(X_live)[0])
    median_temp = raw_quantiles[2] # 50% 分位点
    
    alphas = [0.05, 0.25, 0.5, 0.75, 0.95]
    cdf_interp = interp1d(raw_quantiles, alphas, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    
    target_temps = [int(median_temp)-2, int(median_temp)-1, int(median_temp), int(median_temp)+1, int(median_temp)+2]
    
    probs = []
    for t in target_temps:
        prob = cdf_interp(t + 0.5) - cdf_interp(t - 0.5)
        probs.append({"temp": t, "prob": round(prob * 100, 1)})
        
    probs = sorted(probs, key=lambda x: x['prob'], reverse=True)
    
    # 汇总输出
    output = {
        "update_time": data['update_time'],
        "hour": current_hour,
        "realtime": {
            "current_temp": data['wu_realtime']['temp'],
            "max_temp": data['wu_realtime']['max_temp_so_far'],
            "forecast_mean": round(df_features['forecast_temp_mean'].iloc[0], 2)
        },
        "model": {
            "median": round(median_temp, 2),
            "quantiles": {
                "p05": round(raw_quantiles[0], 2),
                "p95": round(raw_quantiles[4], 2)
            },
            "probabilities": probs
        }
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_bot()
    print(f"数据更新完成，已写入 {OUTPUT_JSON}")