from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .config import TournamentConfig
from .data_sources import fetch_and_stitch
from .features import make_supervised
from .registry import load_registry
from .storage import Storage
from .env import load_env


def _load_latest_dataset(config: TournamentConfig):
    storage = Storage(config.db_path, config.ohlcv_table)
    storage.init_db()
    df = storage.load()
    if df.empty:
        start = datetime.fromisoformat(config.start_date_utc).replace(tzinfo=timezone.utc)
        fetched, _ = fetch_and_stitch(config.symbol, config.yfinance_symbol, start, config.timeframe, config.candle_minutes)
        if not fetched.empty:
            fetched = fetched.set_index("timestamp_utc")
            storage.upsert(fetched)
        df = storage.load()
    return df


def _load_model(model_path: str):
    import joblib
    return joblib.load(model_path)


def _direction_confidence(model, X_row) -> float:
    try:
        proba = model.predict_proba(X_row)[0]
        return float(max(proba))
    except Exception:
        return 0.5


def _resolve_feature_cols(model, fallback_cols):
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)


def risk_label(conf: float, vol_flag: int, drawdown: float) -> str:
    if conf >= 0.65 and vol_flag == 0 and drawdown < 0.15:
        return "Opportunity"
    if conf >= 0.6 and vol_flag == 0:
        return "Neutral"
    if conf < 0.55 or vol_flag == 1 or drawdown >= 0.25:
        return "Avoid"
    return "Cautious"


def predict_latest(config: TournamentConfig) -> Dict[str, str]:
    registry = load_registry(config.registry_path)
    df = _load_latest_dataset(config)
    sup = make_supervised(df, candle_minutes=config.candle_minutes, feature_windows_hours=config.feature_windows)
    if sup.empty:
        raise RuntimeError("No data for prediction")

    latest = sup.iloc[-1:]

    outputs = {}
    for task in ["direction", "return", "range"]:
        champ = registry.get("champions", {}).get(task)
        if not champ:
            outputs[task] = "No champion"
            continue
        model = _load_model(champ["model_path"])
        feature_cols = _resolve_feature_cols(model, champ.get("feature_cols", []))
        X = latest.reindex(columns=feature_cols, fill_value=0.0)
        drawdown = float(champ.get("metrics", {}).get("mdd", 0.0))
        vol_flag = int(latest["high_vol_flag"].values[0]) if "high_vol_flag" in latest else 0

        if task == "direction":
            pred = int(model.predict(X)[0])
            conf = _direction_confidence(model, X)
            label = "UP" if pred == 1 else "DOWN"
            outputs[task] = f"{label} (prob {conf:.2f}) | Risk: {risk_label(conf, vol_flag, drawdown)}"
        elif task == "return":
            pred = float(model.predict(X)[0])
            outputs[task] = f"Expected return: {pred:.4f} | Risk: {risk_label(0.6, vol_flag, drawdown)}"
        else:
            preds = model.predict(X)[0]
            p10, p50, p90 = preds
            outputs[task] = (
                f"Range P10/P50/P90: {p10:.4f}, {p50:.4f}, {p90:.4f} | Risk: {risk_label(0.6, vol_flag, drawdown)}"
            )
    return outputs


def main():
    load_env()
    config = TournamentConfig()
    outputs = predict_latest(config)
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
