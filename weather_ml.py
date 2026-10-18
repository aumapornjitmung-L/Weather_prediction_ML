"""โมดูลสำหรับดึงข้อมูลอากาศย้อนหลัง สร้างฟีเจอร์ เทรนโมเดล และพยากรณ์ล่วงหน้า 1-7 วัน"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

DAILY_VARS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "rain",
    "wind_speed_10m_max": "wind",
    "shortwave_radiation_sum": "radiation",
}
COLS = list(DAILY_VARS.values())
HORIZONS = range(1, 8)  # พยากรณ์ 1-7 วันข้างหน้า
RAIN_MM = 1.0           # ถือว่า "ฝนตก" เมื่อปริมาณฝน >= 1 มม./วัน


def search_city(name: str, count: int = 5) -> list[dict]:
    r = requests.get(
        GEOCODE_URL,
        params={"name": name, "count": count, "language": "th"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_history(lat: float, lon: float, years: int = 8) -> pd.DataFrame:
    end = date.today() - timedelta(days=2)  # ข้อมูล archive ล่าช้าประมาณ 2 วัน
    start = end - timedelta(days=365 * years)
    r = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
        },
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=DAILY_VARS)
    df = df.asfreq("D").interpolate(limit=3).dropna()
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in COLS:
        for lag in range(7):  # lag0 = วันนี้
            f[f"{c}_lag{lag}"] = df[c].shift(lag)
        f[f"{c}_roll14"] = df[c].rolling(14).mean()
        f[f"{c}_roll30"] = df[c].rolling(30).mean()
    doy = df.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def _fit(X, y, classifier=False):
    if classifier:
        if y.nunique() < 2:
            return None
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y)


def run_pipeline(lat: float, lon: float, years: int = 8):
    """คืนค่า (history, forecast, metrics)"""
    df = fetch_history(lat, lon, years)
    X_all = make_features(df)
    last_row = X_all.iloc[[-1]]
    last_date = df.index[-1]

    forecast_rows, metric_rows = [], []
    for h in HORIZONS:
        fut_rain = df["rain"].shift(-h)
        targets = {
            "tmax": df["tmax"].shift(-h),
            "tmin": df["tmin"].shift(-h),
            "rain": (fut_rain >= RAIN_MM).astype(float).where(fut_rain.notna()),
        }
        row = {"date": last_date + pd.Timedelta(days=h), "horizon": h}
        m = {"horizon": h}

        for name, y in targets.items():
            data = X_all.join(y.rename("y")).dropna()
            X, yy = data.drop(columns="y"), data["y"]
            split = int(len(data) * 0.8)  # แบ่งตามเวลา ไม่สุ่ม เพื่อไม่ให้ข้อมูลอนาคตรั่ว
            X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], yy.iloc[:split], yy.iloc[split:]

            if name == "rain":
                clf = _fit(X_tr, y_tr, classifier=True)
                pred = clf.predict(X_te) if clf else np.zeros(len(y_te))
                base = (X_te["rain_lag0"] >= RAIN_MM).astype(float)
                m["rain_acc"] = float((pred == y_te).mean())
                m["rain_acc_baseline"] = float((base == y_te).mean())
                final = _fit(X, yy, classifier=True)
                row["rain_prob"] = float(final.predict_proba(last_row)[0, 1]) if final else 0.0
            else:
                reg = _fit(X_tr, y_tr)
                pred = reg.predict(X_te)
                m[f"{name}_mae"] = float(np.mean(np.abs(pred - y_te)))
                m[f"{name}_mae_baseline"] = float(np.mean(np.abs(X_te[f"{name}_lag0"] - y_te)))
                row[name] = float(_fit(X, yy).predict(last_row)[0])

        forecast_rows.append(row)
        metric_rows.append(m)

    return df, pd.DataFrame(forecast_rows), pd.DataFrame(metric_rows)
