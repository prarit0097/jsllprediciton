from jeena_sikho_dashboard.services import _apply_match_fields, _compute_match_metrics


def test_match_nonzero_diff_not_100():
    metrics = _compute_match_metrics(89233.89, 89222.81)
    assert metrics["abs_diff"] is not None
    assert round(metrics["abs_diff"], 2) == 11.08
    assert metrics["match_percent"] is not None
    assert metrics["match_percent"] < 100.0


def test_match_equal_is_100():
    metrics = _compute_match_metrics(100.0, 100.0)
    assert metrics["abs_diff"] == 0.0
    assert metrics["match_percent"] == 100.0


def test_match_actual_zero_safe():
    metrics = _compute_match_metrics(100.0, 0.0)
    assert metrics["abs_diff"] == 100.0
    assert metrics["match_percent"] is None


def test_match_clamped():
    metrics = _compute_match_metrics(0.0, 1.0)
    assert metrics["match_percent"] == 0.0


def test_match_precise_rounding():
    row = {"predicted_price": 89165.90, "actual_price_1h": 89149.45}
    _apply_match_fields(row)
    assert row["abs_diff_usd"] == 16.45
    assert row["match_percent_precise"] == 99.9815


def test_tiny_diff_not_100():
    metrics = _compute_match_metrics(100.00001, 100.0)
    assert metrics["match_percent"] is not None
    assert metrics["match_percent"] < 100.0
    assert metrics["match_percent"] <= 99.9999

