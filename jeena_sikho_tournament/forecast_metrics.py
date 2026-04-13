from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _safe_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _nan_or_float(value: float) -> Optional[float]:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def summarize_price_forecast(
    actual_price,
    predicted_price,
    *,
    actual_return=None,
    predicted_return=None,
    lower_price=None,
    upper_price=None,
) -> Dict[str, Any]:
    actual_price_arr = _safe_array(actual_price)
    predicted_price_arr = _safe_array(predicted_price)
    if actual_price_arr.size == 0 or predicted_price_arr.size == 0 or actual_price_arr.size != predicted_price_arr.size:
        return {
            "sample_count": 0,
            "price_mae": None,
            "price_rmse": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "mape": None,
            "smape": None,
            "signed_bias_rs": None,
            "direction_hit_rate": None,
            "band_80_coverage": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "return_mae": None,
            "avg_actual_price": None,
        }

    abs_err = np.abs(predicted_price_arr - actual_price_arr)
    price_mae = float(np.mean(abs_err))
    price_rmse = float(np.sqrt(np.mean(np.square(predicted_price_arr - actual_price_arr))))
    median_abs_error = float(np.median(abs_err))
    p90_abs_error = float(np.quantile(abs_err, 0.9))

    denom = np.where(np.abs(actual_price_arr) > 0, np.abs(actual_price_arr), np.nan)
    mape = float(np.nanmean((abs_err / denom) * 100.0))
    smape_denom = (np.abs(actual_price_arr) + np.abs(predicted_price_arr)) / 2.0
    smape = float(np.nanmean(np.where(smape_denom > 0, abs_err / smape_denom, np.nan)) * 100.0)

    out: Dict[str, Any] = {
        "sample_count": int(actual_price_arr.size),
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "median_abs_error": median_abs_error,
        "p90_abs_error": p90_abs_error,
        "mape": _nan_or_float(mape),
        "smape": _nan_or_float(smape),
        "signed_bias_rs": float(np.mean(predicted_price_arr - actual_price_arr)),
        "direction_hit_rate": None,
        "band_80_coverage": None,
        "calibration_slope": None,
        "calibration_intercept": None,
        "return_mae": None,
        "avg_actual_price": float(np.mean(np.abs(actual_price_arr))),
    }

    if actual_return is not None and predicted_return is not None:
        actual_return_arr = _safe_array(actual_return)
        predicted_return_arr = _safe_array(predicted_return)
        if actual_return_arr.size == predicted_return_arr.size == actual_price_arr.size:
            out["return_mae"] = float(np.mean(np.abs(actual_return_arr - predicted_return_arr)))
            hit = (np.sign(actual_return_arr) == np.sign(predicted_return_arr)).astype(float)
            out["direction_hit_rate"] = float(np.mean(hit) * 100.0)
            try:
                X = np.column_stack([np.ones(actual_return_arr.size), predicted_return_arr])
                coeff, *_ = np.linalg.lstsq(X, actual_return_arr, rcond=None)
                out["calibration_intercept"] = float(coeff[0])
                out["calibration_slope"] = float(coeff[1])
            except Exception:
                out["calibration_intercept"] = None
                out["calibration_slope"] = None

    if lower_price is not None and upper_price is not None:
        lower_arr = _safe_array(lower_price)
        upper_arr = _safe_array(upper_price)
        if lower_arr.size == upper_arr.size == actual_price_arr.size:
            coverage = ((actual_price_arr >= lower_arr) & (actual_price_arr <= upper_arr)).astype(float)
            out["band_80_coverage"] = float(np.mean(coverage))

    return out
