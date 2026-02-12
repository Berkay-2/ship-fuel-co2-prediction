# run_thesis_models.py
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import joblib

warnings.filterwarnings("ignore")


FUEL_CSV = "ship_fuel_efficiency.csv"
SHIP_PERF_CSV = "Ship_Performance_Dataset.csv"
ENGINE_FAULT_CSV = "marine_engine_fault_dataset .csv"

OUT_DIR = Path("models_out")
OUT_DIR.mkdir(exist_ok=True)


def resolve_csv_path(preferred: str) -> Optional[Path]:
    """Return usable dataset path, tolerating renamed CSV downloads."""
    path = Path(preferred)
    if path.exists():
        return path

    stem = path.stem
    matches = sorted(Path(".").glob(f"{stem}*.csv"))
    if matches:
        resolved = matches[0]
        print(f"[INFO] Resolved missing {preferred!r} to {resolved.name!r}")
        return resolved

    print(f"Missing file: {preferred}")
    return None


def load_csv(preferred: str) -> Optional[pd.DataFrame]:
    """Read dataset after resolving renamed download files."""
    path = resolve_csv_path(preferred)
    if path is None:
        return None
    return pd.read_csv(path)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ]
    )


def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_regression_models(df: pd.DataFrame, target: str, drop_cols: list[str], tag: str):
    df = df.copy()
    df = df.dropna(subset=[target])

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    models["Ridge"] = {"model": Ridge(alpha=1.0, random_state=42), "scale": True}

    models["RandomForest"] = {
        "model": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ),
        "scale": False
    }

    # optional: XGBoost, LightGBM, CatBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = {
            "model": XGBRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            ),
            "scale": False
        }
    except Exception:
        pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = {
            "model": lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            ),
            "scale": False
        }
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = {
            "model": CatBoostRegressor(
                iterations=1500,
                learning_rate=0.03,
                depth=8,
                loss_function="RMSE",
                random_seed=42,
                verbose=False
            ),
            "scale": False
        }
    except Exception:
        pass

    rows = []
    best = None

    for name, cfg in models.items():
        pre = make_preprocessor(X_train, scale_numeric=cfg["scale"])
        pipe = Pipeline([("preprocess", pre), ("model", cfg["model"])])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae, rmse, r2 = reg_metrics(y_test, pred)
        rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

        if best is None or mae < best["MAE"]:
            best = {"name": name, "MAE": mae, "pipe": pipe}

    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    print("\n==============================")
    print(f"REGRESSION RESULTS | {tag} | target={target}")
    print("==============================")
    print(res)

    out_csv = OUT_DIR / f"{tag}_{target}_results.csv"
    res.to_csv(out_csv, index=False)

    best_path = OUT_DIR / f"{tag}_{target}_BEST_{best['name']}.joblib"
    joblib.dump(best["pipe"], best_path)

    print(f"\nSaved results: {out_csv}")
    print(f"Saved best model: {best_path}")

    return res


def train_fault_classifier(df: pd.DataFrame, target: str = "Fault_Label", tag: str = "engine_fault"):
    df = df.copy()
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=700, random_state=42, n_jobs=-1
    )

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
    except Exception:
        pass

    best = None
    rows = []

    for name, model in models.items():
        pre = make_preprocessor(X_train, scale_numeric=False)
        pipe = Pipeline([("preprocess", pre), ("model", model)])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        rows.append({"model": name, "accuracy": acc, "f1_weighted": f1})

        if best is None or f1 > best["f1"]:
            best = {"name": name, "f1": f1, "pipe": pipe}

    res = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    print("\n==============================")
    print(f"CLASSIFICATION RESULTS | {tag} | target={target}")
    print("==============================")
    print(res)

    best_pipe = best["pipe"]
    best_pred = best_pipe.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred))

    out_csv = OUT_DIR / f"{tag}_results.csv"
    res.to_csv(out_csv, index=False)

    best_path = OUT_DIR / f"{tag}_BEST_{best['name']}.joblib"
    joblib.dump(best_pipe, best_path)

    print(f"\nSaved results: {out_csv}")
    print(f"Saved best model: {best_path}")

    return res


def main():
    # 1) Fuel + CO2 dataset
    df_fuel = load_csv(FUEL_CSV)
    if df_fuel is not None:

        # fuel_consumption and CO2_emissions are the targets in this dataset
        base_drop = ["fuel_consumption", "CO2_emissions"]

        train_regression_models(
            df=df_fuel,
            target="fuel_consumption",
            drop_cols=base_drop,
            tag="fuel_dataset"
        )

        train_regression_models(
            df=df_fuel,
            target="CO2_emissions",
            drop_cols=base_drop,
            tag="fuel_dataset"
        )
    # 2) Ship performance dataset (secondary) - no fuel target here
    # This part is optional, it shows how you can build another regression experiment
    df_ship = load_csv(SHIP_PERF_CSV)
    if df_ship is not None:

        if "Operational_Cost_USD" in df_ship.columns:
            train_regression_models(
                df=df_ship,
                target="Operational_Cost_USD",
                drop_cols=["Operational_Cost_USD"],
                tag="ship_performance"
            )
        else:
            print("Ship_Performance_Dataset.csv loaded, but Operational_Cost_USD not found.")
    # 3) Engine fault dataset
    df_fault = load_csv(ENGINE_FAULT_CSV)
    if df_fault is not None:
        if "Fault_Label" in df_fault.columns:
            train_fault_classifier(df_fault, target="Fault_Label", tag="engine_fault")
        else:
            print("Fault_Label column not found in engine fault dataset.")


if __name__ == "__main__":
    main()
