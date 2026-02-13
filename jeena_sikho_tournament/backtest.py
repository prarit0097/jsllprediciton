import numpy as np


def _equity_curve(positions, log_returns, fee):
    simple_ret = np.expm1(log_returns)
    equity = [1.0]
    for pos, r in zip(positions, simple_ret):
        step_ret = pos * r - (fee * pos)
        equity.append(equity[-1] * (1 + step_ret))
    return np.array(equity[1:])


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(np.abs(dd.min()))


def trading_score(positions, log_returns, fee) -> tuple:
    equity = _equity_curve(positions, log_returns, fee)
    if len(equity) == 0:
        return 0.0, 0.0, 0.0
    net = equity[-1] - 1.0
    mdd = max_drawdown(equity)
    score = (0.5 + 0.5 * np.tanh(net * 5)) * (1 - mdd)
    return float(net), float(mdd), float(score)
