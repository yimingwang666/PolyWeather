import os
import re
import json
import math
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from scipy.interpolate import interp1d
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

# === Logging Configuration (Optimized for Daemon/Cronjob) ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# === Global Constants ===
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
MODELS_DIR = "models"
DATA_DIR = "data"
CONFIG_FILE = "config.js"

MODEL_NAMES = {
    'temperature_2m_ecmwf_ifs025': 'ECMWF (EU)',
    'temperature_2m_gfs_seamless': 'GFS (US)',
    'temperature_2m_icon_seamless': 'ICON (DE)',
    'temperature_2m_jma_seamless': 'JMA (JP)',
    'temperature_2m_ukmo_seamless': 'UKMO (UK)',
    'temperature_2m_cma_grapes_global': 'CMA (CN)'
}

# === Core Utility Functions ===
def clean_float(val):
    return float(round(val, 2)) if val is not None and not pd.isna(val) and not math.isnan(val) else None

def load_config():
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Config file {CONFIG_FILE} not found.")
        return {}
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        match = re.search(r'const\s+CITY_CONFIG\s*=\s*(\{.*?\});', f.read(), re.DOTALL)
        if match:
            clean_json = re.sub(r',\s*}', '}', match.group(1))
            return json.loads(clean_json)
    logger.error("Failed to parse config.js.")
    return {}

# === Data Ingestion Pipeline ===
def fetch_data(city_id, cfg, local_time):
    today = local_time.strftime('%Y%m%d')
    om_today = local_time.strftime('%Y-%m-%d')
    unit_wu = "e" if cfg.get('unit') == "F" else "m"
    
    # 1. Fetch Wunderground (Live Actuals)
    wu_data = {"temp": np.nan, "max_temp_so_far": np.nan, "rh": np.nan, "wdir": np.nan, "wspd": np.nan, "pressure": np.nan}
    actual_temp_24h = [None] * 24
    wu_success = False

    wu_url = f"https://api.weather.com/v1/location/{cfg['wu_code']}/observations/historical.json?apiKey={WU_API_KEY}&units={unit_wu}&startDate={today}&endDate={today}"
    try:
        resp = requests.get(wu_url, timeout=10)
        if resp.status_code == 200 and 'observations' in resp.json():
            df_wu = pd.DataFrame(resp.json()['observations'])
            if not df_wu.empty:
                latest = df_wu.iloc[-1]
                wu_data.update({
                    "max_temp_so_far": df_wu['temp'].max(),
                    "temp": latest.get('temp', np.nan), "rh": latest.get('rh', np.nan),
                    "wdir": latest.get('wdir', np.nan), "wspd": latest.get('wspd', np.nan),
                    "pressure": latest.get('pressure', np.nan)
                })
                df_wu['datetime'] = pd.to_datetime(df_wu['valid_time_gmt'], unit='s', utc=True).dt.tz_convert(cfg['tz'])
                hourly_actual = df_wu.groupby(df_wu['datetime'].dt.hour)['temp'].mean().to_dict()
                actual_temp_24h = [clean_float(hourly_actual.get(h)) if h <= local_time.hour else None for h in range(24)]
                wu_success = True
    except Exception as e:
        logger.warning(f"{city_id} WU fetch failed: {e}")

    # 2. Fetch Open-Meteo (Forecasts & Fallback Actuals)
    tz_enc = cfg['tz'].replace('/', '%2F')
    
    # Extract dynamic models string for correct OM API syntax
    models_str = ",".join([k.replace('temperature_2m_', '') for k in MODEL_NAMES.keys()])
    
    om_url = (f"https://api.open-meteo.com/v1/forecast?latitude={cfg['lat']}&longitude={cfg['lon']}"
              f"&hourly=temperature_2m&models={models_str}"
              f"&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
              f"&timezone={tz_enc}&forecast_days=2")
              
    if cfg.get('unit') == "F":
        om_url += "&temperature_unit=fahrenheit&wind_speed_unit=mph"

    try:
        om_resp = requests.get(om_url, timeout=15).json()
        
        # Intercept OM specific API errors gracefully
        if 'error' in om_resp:
            logger.error(f"{city_id} OM API Error: {om_resp.get('reason')}")
            return None, None, None
            
        df_om = pd.DataFrame(om_resp['hourly'])
        df_om['time'] = pd.to_datetime(df_om['time'])
        
        # OM Fallback mechanism if WU fails
        if not wu_success:
            cur = om_resp.get('current', {})
            wu_data.update({
                "temp": cur.get('temperature_2m', np.nan), "rh": cur.get('relative_humidity_2m', np.nan),
                "pressure": cur.get('surface_pressure', np.nan), "wspd": cur.get('wind_speed_10m', np.nan),
                "wdir": cur.get('wind_direction_10m', np.nan)
            })
            df_past = df_om[(df_om['time'].dt.strftime('%Y-%m-%d') == om_today) & (df_om['time'].dt.hour <= local_time.hour)]
            if not df_past.empty:
                wu_data['max_temp_so_far'] = max(wu_data['temp'], df_past[list(MODEL_NAMES.keys())[0]].max())
                
        return wu_data, actual_temp_24h, df_om
    except Exception as e:
        logger.error(f"{city_id} OM fetch failed: {e}")
        return None, None, None

# === Feature Engineering & Model Inference ===
def build_inference(model, target_date, is_tomorrow, wu_data, df_om, local_time, actual_temp_24h):
    df_day = df_om[df_om['time'].dt.strftime('%Y-%m-%d') == target_date].copy()
    if df_day.empty: return None
    
    daily_maxes = {f"daily_max_forecast_{m.replace('temperature_2m_', '')}": df_day[m].max() for m in MODEL_NAMES.keys()}

    if is_tomorrow:
        target_hour = 0
        hour_0_slice = df_day[df_day['time'].dt.hour == 0].iloc[0]
        current_temp = max_so_far = hour_0_slice[list(MODEL_NAMES.keys())[0]]
        rh = wdir = wspd = pressure = 0
    else:
        target_hour = local_time.hour
        current_temp, max_so_far = wu_data['temp'], wu_data['max_temp_so_far']
        rh, wdir, wspd, pressure = wu_data['rh'], wu_data['wdir'], wu_data['wspd'], wu_data['pressure']

    hour_slice = df_day[df_day['time'].dt.hour == target_hour].iloc[0]
    features = {'temp': current_temp, 'max_temp_so_far': max_so_far, 'rh': rh, 'wdir': wdir, 'wspd': wspd, 'pressure': pressure}
    
    model_preds = [hour_slice[m] for m in MODEL_NAMES.keys() if not pd.isna(hour_slice[m])]
    features.update({m: hour_slice[m] for m in MODEL_NAMES.keys()})
    features['forecast_temp_mean'] = np.mean(model_preds) if model_preds else np.nan
    features['forecast_temp_std'] = np.std(model_preds) if model_preds else np.nan
    features.update(daily_maxes)
    
    dt_obj = datetime.strptime(target_date, '%Y-%m-%d')
    features.update({
        'hour': target_hour, 'month': dt_obj.month,
        'hour_sin': np.sin(2 * np.pi * target_hour / 24), 'hour_cos': np.cos(2 * np.pi * target_hour / 24),
        'month_sin': np.sin(2 * np.pi * dt_obj.month / 12), 'month_cos': np.cos(2 * np.pi * dt_obj.month / 12)
    })
    
    df_x = pd.DataFrame([features]).fillna(0)
    for c in model.feature_names_:
        if c not in df_x.columns: df_x[c] = 0.0
    
    # === Execute Quantile Regression ===
    quantiles = np.sort(model.predict(df_x[model.feature_names_])[0])
    
    if not is_tomorrow and not pd.isna(max_so_far):
        quantiles = np.maximum(quantiles, max_so_far)
        
    median = quantiles[2]
    
    # CDF Integration for Pricing
    cdf = interp1d(quantiles, [0.05, 0.25, 0.5, 0.75, 0.95], kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    targets = [int(median)-2, int(median)-1, int(median), int(median)+1, int(median)+2]
    probs = sorted([{"temp": t, "prob": clean_float((cdf(t+0.5) - cdf(t-0.5)) * 100)} for t in targets], key=lambda x: x['prob'], reverse=True)

    inst_res, max_vals = [], []
    for k, v in MODEL_NAMES.items():
        val = daily_maxes[f"daily_max_forecast_{k.replace('temperature_2m_', '')}"]
        inst_res.append({"name": v, "temp": clean_float(val)})
        if not pd.isna(val): max_vals.append(val)

    return {
        "date": target_date,
        "realtime": {"current_temp": clean_float(current_temp), "max_temp": clean_float(max_so_far), "forecast_mean": clean_float(np.mean(max_vals)) if max_vals else "N/A", "forecast_std": clean_float(np.std(max_vals)) if max_vals else "N/A"},
        "institutions": inst_res,
        "chart_data": {"hours": [f"{i:02d}:00" for i in range(24)], "actual_temp": actual_temp_24h if not is_tomorrow else [None]*24, "forecasts": {v: [clean_float(x) for x in df_day[k].values] for k, v in MODEL_NAMES.items()}},
        "model": {"median": clean_float(median), "quantiles": {"p05": clean_float(quantiles[0]), "p25": clean_float(quantiles[1]), "p50": clean_float(quantiles[2]), "p75": clean_float(quantiles[3]), "p95": clean_float(quantiles[4])}, "probabilities": probs}
    }

# === Master Orchestrator ===
def process_city(city_id, cfg):
    model_path = os.path.join(MODELS_DIR, f"{city_id}_model.cbm")
    if not os.path.exists(model_path): return
    
    tz = pytz.timezone(cfg['tz'])
    now = datetime.now(tz)
    wu_data, actual_24h, df_om = fetch_data(city_id, cfg, now)
    if df_om is None: return
    
    model = CatBoostRegressor().load_model(model_path)
    res_today = build_inference(model, now.strftime('%Y-%m-%d'), False, wu_data, df_om, now, actual_24h)
    res_tomorrow = build_inference(model, (now + timedelta(days=1)).strftime('%Y-%m-%d'), True, wu_data, df_om, now, actual_24h)
    
    if res_today and res_tomorrow:
        output_file = os.path.join(DATA_DIR, f"{city_id}_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"update_time": datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'), "hour": now.hour, "today": res_today, "tomorrow": res_tomorrow}, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated {city_id.upper()}")

if __name__ == "__main__":
    logger.info("Initializing PolyWeather Inference Engine...")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created missing directory: {MODELS_DIR}/")
        
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created missing directory: {DATA_DIR}/")

    configs = load_config()
    for cid, cfg in configs.items():
        try:
            process_city(cid, cfg)
        except Exception as e:
            logger.error(f"Critical error processing {cid}: {e}")
            
    logger.info("Routine complete.")
