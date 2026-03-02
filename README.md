# PolyWeather

**Quantitative Weather Prediction Terminal for Polymarket**

PolyWeather is a fully automated quantitative prediction terminal designed specifically for the Polymarket weather derivatives market. By aggregating multi-source meteorological data and leveraging the CatBoost machine learning algorithm, the system generates probability distributions for daily maximum temperatures to assist in trading decisions.

## 📊 Core Features

* **Multi-Model Ensemble**: Real-time aggregation of forecast data from 6 major global meteorological agencies (ECMWF, GFS, ICON, JMA, UKMO, CMA) via the Open-Meteo API.
* **Quantile Regression**: Utilizes the CatBoost algorithm under the hood to output the predicted median and key quantiles (5%, 25%, 50%, 75%, 95%), effectively capturing tail probabilities and extreme weather risks.
* **Real-Time Calibration**: Integrates live data from Wunderground airport observation stations to dynamically correct forecast deviations during in-play market hours.
* **Serverless Automation**: Implements a fully automated scheduled pipeline via GitHub Actions, covering data scraping, model inference, and frontend rendering.

## 🌍 Market Coverage

The system is currently live for Seoul and is actively expanding to cover other major global weather markets on Polymarket.

| City | Country/Region | ICAO Code | Unit | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Seoul** | 🇰🇷 South Korea | `RKSI` | Celsius (°C) | 🟢 Active |
| **Miami** | 🇺🇸 United States | `KMIA` | Fahrenheit (°F) | 🟡 Planned |
| **London** | 🇬🇧 United Kingdom | `EGLC` | Celsius (°C) | 🟡 Planned |
| **Wellington** | 🇳🇿 New Zealand | `NZWN` | Celsius (°C) | 🟡 Planned |
| **Buenos Aires** | 🇦🇷 Argentina | `SAEZ` | Celsius (°C) | 🟡 Planned |
| **Toronto** | 🇨🇦 Canada | `CYYZ` | Celsius (°C) | 🟡 Planned |
| **Ankara** | 🇹🇷 Turkey | `LTAC` | Celsius (°C) | 🟡 Planned |
| **Paris** | 🇫🇷 France | `LFPG` | Celsius (°C) | 🟡 Planned |
| **Sao Paulo** | 🇧🇷 Brazil | `SBGR` | Celsius (°C) | 🟡 Planned |
| **Chicago** | 🇺🇸 United States | `KORD` | Fahrenheit (°F) | 🟡 Planned |

*(Note: For markets involving U.S. airports, the underlying API scraping and model training have been adapted for Imperial units and Fahrenheit conversions.)*

## ⚙️ System Architecture

1. **Backend Inference (`run_bot.py`)**:
   * Fetches the latest meteorological features via API.
   * Loads the pre-trained `.cbm` decision tree model for inference.
   * Calculates the continuous Cumulative Distribution Function (CDF) for target temperature ranges and exports the results to `data.json`.

2. **Frontend Dashboard (`index.html`)**:
   * A lightweight static data dashboard built with pure HTML/JS and Chart.js.
   * Visually displays inter-agency forecast divergence (standard deviation), probability distribution histograms, and 24-hour actual vs. forecasted temperature trend lines.

3. **Automated Scheduling (`.github/workflows/quant.yml`)**:
   * Periodically triggers data updates and page rebuilds via a Cron timer.

## ⚠️ Disclaimer

This project is intended strictly for data analysis and machine learning technical exchange. Meteorological systems exhibit highly chaotic characteristics; the probability distributions output by the model do not constitute financial or investment advice. Please independently evaluate the liquidity and settlement rules of Polymarket before trading.
