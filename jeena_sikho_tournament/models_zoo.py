from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ModelSpec:
    name: str
    model: Any
    task: str
    meta: Dict[str, Any]


class NaiveLastReturn:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(X["ret_1c"].values > 0, 1, 0)


class EMABaseline:
    def __init__(self, span: int = 12):
        self.span = span

    def fit(self, X, y):
        return self

    def predict(self, X):
        ema = X["ret_1c"].ewm(span=self.span, adjust=False).mean()
        return np.where(ema.values > 0, 1, 0)


class LogisticBaseline:
    def fit(self, X, y):
        self.bias = int(np.mean(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(len(X), self.bias)


class ZeroRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class QuantileBundle:
    def __init__(self, base_models: Dict[float, Any]):
        self.base_models = base_models

    def fit(self, X, y):
        for q, model in self.base_models.items():
            model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        preds = []
        for q in sorted(self.base_models.keys()):
            preds.append(self.base_models[q].predict(X))
        return np.vstack(preds).T


def _sklearn_candidates(task: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    try:
        from sklearn.linear_model import (
            LogisticRegression,
            SGDClassifier,
            SGDRegressor,
            RidgeClassifier,
            Ridge,
            Lasso,
            ElasticNet,
        )
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import LinearSVR
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            AdaBoostClassifier,
            AdaBoostRegressor,
        )
    except Exception:
        return specs

    if task == "direction":
        for c in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            specs.append(ModelSpec(f"logreg_l2_c{c}", LogisticRegression(max_iter=1000, C=c), task, {"family": "logreg", "group": "fast"}))
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
            specs.append(ModelSpec(f"ridge_clf_a{alpha}", RidgeClassifier(alpha=alpha), task, {"family": "ridge", "group": "fast"}))
        for n in [3, 5, 9, 15, 21, 31]:
            specs.append(ModelSpec(f"knn_{n}", KNeighborsClassifier(n_neighbors=n, weights="distance"), task, {"family": "knn", "group": "fast"}))
        for n, lr in [(200, 0.5), (400, 0.3), (600, 0.2), (800, 0.1)]:
            specs.append(ModelSpec(f"ada_clf_{n}_{lr}", AdaBoostClassifier(n_estimators=n, learning_rate=lr), task, {"family": "ada", "group": "fast"}))
        for alpha in [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
            specs.append(ModelSpec(f"sgd_log_a{alpha}", SGDClassifier(loss="log_loss", alpha=alpha), task, {"family": "sgd", "group": "fast"}))
        for n, d in [
            (200, None),
            (300, None),
            (400, None),
            (400, 6),
            (600, 8),
            (800, 10),
            (1000, 12),
            (1200, None),
            (800, 6),
        ]:
            specs.append(ModelSpec(f"rf_{n}_{d}", RandomForestClassifier(n_estimators=n, max_depth=d), task, {"family": "rf", "group": "medium"}))
        for n, d in [
            (200, None),
            (400, None),
            (600, None),
            (400, 8),
            (600, 10),
            (800, 12),
            (1000, None),
            (800, 8),
        ]:
            specs.append(ModelSpec(f"et_{n}_{d}", ExtraTreesClassifier(n_estimators=n, max_depth=d), task, {"family": "et", "group": "medium"}))
        for n, lr in [(100, 0.1), (200, 0.1), (300, 0.05), (400, 0.05), (500, 0.03), (600, 0.03), (700, 0.03), (800, 0.02)]:
            specs.append(ModelSpec(f"gb_{n}_{lr}", GradientBoostingClassifier(n_estimators=n, learning_rate=lr), task, {"family": "gb", "group": "fast"}))
        for n in [100, 200, 300, 400, 500, 600, 700, 800]:
            specs.append(ModelSpec(f"hgb_{n}", HistGradientBoostingClassifier(max_iter=n), task, {"family": "hgb", "group": "fast"}))

    if task == "return":
        for alpha in [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
            specs.append(ModelSpec(f"sgd_reg_a{alpha}", SGDRegressor(alpha=alpha), task, {"family": "sgd", "group": "fast"}))
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
            specs.append(ModelSpec(f"ridge_reg_a{alpha}", Ridge(alpha=alpha), task, {"family": "ridge", "group": "fast"}))
        for alpha in [1e-4, 5e-4, 1e-3]:
            specs.append(ModelSpec(f"lasso_a{alpha}", Lasso(alpha=alpha, max_iter=2000), task, {"family": "lasso", "group": "fast"}))
        for alpha, l1 in [(1e-4, 0.2), (5e-4, 0.5), (1e-3, 0.7)]:
            specs.append(ModelSpec(f"enet_a{alpha}_l1{l1}", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=2000), task, {"family": "enet", "group": "fast"}))
        for n in [3, 5, 9, 15, 21, 31]:
            specs.append(ModelSpec(f"knn_reg_{n}", KNeighborsRegressor(n_neighbors=n, weights="distance"), task, {"family": "knn", "group": "fast"}))
        for c in [0.5, 1.0, 2.0]:
            specs.append(ModelSpec(f"lsvr_c{c}", LinearSVR(C=c, epsilon=0.0005, max_iter=10000), task, {"family": "svr", "group": "fast"}))
        for n, lr in [(200, 0.5), (400, 0.3), (600, 0.2), (800, 0.1)]:
            specs.append(ModelSpec(f"ada_reg_{n}_{lr}", AdaBoostRegressor(n_estimators=n, learning_rate=lr), task, {"family": "ada", "group": "fast"}))
        for n, d in [
            (200, None),
            (300, None),
            (400, None),
            (400, 6),
            (600, 8),
            (800, 10),
            (1000, 12),
            (1200, None),
            (800, 6),
        ]:
            specs.append(ModelSpec(f"rf_reg_{n}_{d}", RandomForestRegressor(n_estimators=n, max_depth=d), task, {"family": "rf", "group": "medium"}))
        for n, d in [
            (200, None),
            (400, None),
            (600, None),
            (400, 8),
            (600, 10),
            (800, 12),
            (1000, None),
            (800, 8),
        ]:
            specs.append(ModelSpec(f"et_reg_{n}_{d}", ExtraTreesRegressor(n_estimators=n, max_depth=d), task, {"family": "et", "group": "medium"}))
        for n, lr in [(200, 0.1), (300, 0.1), (400, 0.05), (500, 0.05), (600, 0.03), (700, 0.03), (800, 0.02)]:
            specs.append(ModelSpec(f"gbr_{n}_{lr}", GradientBoostingRegressor(n_estimators=n, learning_rate=lr), task, {"family": "gb", "group": "fast"}))
        for n in [200, 300, 400, 500, 600, 700, 800]:
            specs.append(ModelSpec(f"hgb_{n}", HistGradientBoostingRegressor(max_iter=n), task, {"family": "hgb", "group": "fast"}))

    if task == "range":
        for n in [50, 80, 120, 160, 200, 240, 300, 360, 420, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]:
            specs.append(ModelSpec(f"gbr_q_{n}", GradientBoostingRegressor(n_estimators=n), task, {"quantile": True, "family": "gbr_q", "group": "fast"}))
        for n in [50, 80, 120, 160, 200, 240, 300, 360, 420, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]:
            specs.append(ModelSpec(f"hgb_q_{n}", HistGradientBoostingRegressor(max_iter=n), task, {"quantile": True, "family": "hgb_q", "group": "fast"}))
    return specs


def _optional_boosters(task: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if task == "direction":
        try:
            import xgboost as xgb  # type: ignore
            for n, d, lr in [
                (200, 3, 0.1),
                (400, 4, 0.05),
                (600, 5, 0.05),
                (800, 6, 0.03),
            ]:
                specs.append(
                    ModelSpec(
                        f"xgb_clf_{n}_d{d}_lr{lr}",
                        xgb.XGBClassifier(
                            n_estimators=n,
                            max_depth=d,
                            learning_rate=lr,
                            subsample=0.8,
                            colsample_bytree=0.8,
                        ),
                        task,
                        {"family": "xgb", "group": "fast"},
                    )
                )
        except Exception:
            pass
        try:
            import lightgbm as lgb  # type: ignore
            for n, lr, leaves in [
                (200, 0.1, 31),
                (400, 0.05, 31),
                (600, 0.05, 63),
                (800, 0.03, 63),
            ]:
                specs.append(
                    ModelSpec(
                        f"lgb_clf_{n}_lr{lr}_l{leaves}",
                        lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves),
                        task,
                        {"family": "lgb", "group": "fast"},
                    )
                )
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier  # type: ignore
            for n, d, lr in [(300, 6, 0.1), (600, 8, 0.05), (900, 10, 0.03)]:
                specs.append(
                    ModelSpec(
                        f"cat_clf_{n}_d{d}_lr{lr}",
                        CatBoostClassifier(iterations=n, depth=d, learning_rate=lr, verbose=False),
                        task,
                        {"family": "cat", "group": "medium"},
                    )
                )
        except Exception:
            pass
    if task == "return":
        try:
            import xgboost as xgb  # type: ignore
            for n, d, lr in [
                (200, 3, 0.1),
                (400, 4, 0.05),
                (600, 5, 0.05),
                (800, 6, 0.03),
            ]:
                specs.append(
                    ModelSpec(
                        f"xgb_reg_{n}_d{d}_lr{lr}",
                        xgb.XGBRegressor(
                            n_estimators=n,
                            max_depth=d,
                            learning_rate=lr,
                            subsample=0.8,
                            colsample_bytree=0.8,
                        ),
                        task,
                        {"family": "xgb", "group": "fast"},
                    )
                )
        except Exception:
            pass
        try:
            import lightgbm as lgb  # type: ignore
            for n, lr, leaves in [
                (200, 0.1, 31),
                (400, 0.05, 31),
                (600, 0.05, 63),
                (800, 0.03, 63),
            ]:
                specs.append(
                    ModelSpec(
                        f"lgb_reg_{n}_lr{lr}_l{leaves}",
                        lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, num_leaves=leaves),
                        task,
                        {"family": "lgb", "group": "fast"},
                    )
                )
        except Exception:
            pass
        try:
            from catboost import CatBoostRegressor  # type: ignore
            for n, d, lr in [(300, 6, 0.1), (600, 8, 0.05), (900, 10, 0.03)]:
                specs.append(
                    ModelSpec(
                        f"cat_reg_{n}_d{d}_lr{lr}",
                        CatBoostRegressor(iterations=n, depth=d, learning_rate=lr, verbose=False),
                        task,
                        {"family": "cat", "group": "medium"},
                    )
                )
        except Exception:
            pass
    if task == "range":
        try:
            import lightgbm as lgb  # type: ignore
            for n, lr, leaves in [
                (200, 0.1, 31),
                (400, 0.05, 31),
                (600, 0.05, 63),
                (800, 0.03, 63),
                (1000, 0.03, 63),
            ]:
                specs.append(
                    ModelSpec(
                        f"lgb_q_{n}_lr{lr}_l{leaves}",
                        lgb.LGBMRegressor(objective="quantile", n_estimators=n, learning_rate=lr, num_leaves=leaves),
                        task,
                        {"quantile": True, "family": "lgb_q", "group": "fast"},
                    )
                )
        except Exception:
            pass
    return specs


def _optional_dl(task: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    try:
        from sklearn.neural_network import MLPClassifier, MLPRegressor
    except Exception:
        return specs

    if task == "direction":
        specs.append(ModelSpec("dl_mlp_clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200), task, {"family": "dl", "group": "heavy"}))
    if task in {"return", "range"}:
        specs.append(ModelSpec("dl_mlp_reg", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200), task, {"family": "dl", "group": "heavy"}))
    return specs


def get_candidates(task: str, max_candidates: int, enable_dl: bool) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    if task == "direction":
        specs.extend(
            [
                ModelSpec("naive_last", NaiveLastReturn(), task, {"baseline": True, "family": "naive", "group": "fast", "required_features": ["ret_1c"]}),
                ModelSpec("ema_12", EMABaseline(span=12), task, {"baseline": True, "family": "ema", "group": "fast", "required_features": ["ret_1c"]}),
                ModelSpec("ema_24", EMABaseline(span=24), task, {"baseline": True, "family": "ema", "group": "fast", "required_features": ["ret_1c"]}),
                ModelSpec("logistic_bias", LogisticBaseline(), task, {"baseline": True, "family": "bias", "group": "fast"}),
            ]
        )
    if task == "return":
        specs.extend(
            [
                ModelSpec("naive_zero", ZeroRegressor(), task, {"baseline": True, "family": "zero", "group": "fast"}),
            ]
        )

    specs.extend(_sklearn_candidates(task))
    specs.extend(_optional_boosters(task))
    if enable_dl:
        specs.extend(_optional_dl(task))

    return specs[:max_candidates]


def _parse_suffix_int(name: str, default: int) -> int:
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return default


def build_quantile_bundle(spec: ModelSpec, quantiles: Tuple[float, float, float]):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    n_estimators = _parse_suffix_int(spec.name, 200)
    q_models = {}
    for q in quantiles:
        if spec.name.startswith("lgb_q"):
            import lightgbm as lgb  # type: ignore

            q_models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators)
        elif spec.name.startswith("hgb_q"):
            q_models[q] = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=n_estimators)
        else:
            q_models[q] = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=n_estimators)
    return QuantileBundle(q_models)
