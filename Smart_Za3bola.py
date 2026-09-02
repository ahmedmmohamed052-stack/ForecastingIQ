import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

from config import settings

# ── Grid size ────────────────────────────────────────────────────────────
# The full grid (default) is meant for a real training run: 3 lag options x
# 3 rolling options x (Ridge grid + XGBoost's 1440-combo grid) x 5 CV folds.
# That's fine on a server but can take a very long time on a laptop.
#
# Set QUICK_TRAIN=true in .env for local development/testing — it shrinks
# every option down to a handful of combinations so /train finishes in
# seconds instead of minutes, at the cost of a less-tuned model. Leave it
# unset (or false) for production-quality training.
if settings.QUICK_TRAIN:
    LAG_OPTIONS     = [8]
    ROLLING_OPTIONS = [3]
else:
    LAG_OPTIONS     = [8, 10, 12]
    ROLLING_OPTIONS = [3, 6, 9]

MODELS = {
    "ridge": {
        "model": Ridge(),
        "params": {
            "model__alpha": [0.1, 1.0] if settings.QUICK_TRAIN else [0.1, 0.5, 1.0, 1.5]
        }
    },
    "xgb": {
        "model": XGBRegressor(random_state=42, objective="reg:squarederror"),
        "params": (
            {
                "model__n_estimators":     [50],
                "model__max_depth":        [4],
                "model__learning_rate":    [0.1],
                "model__subsample":        [0.8],
                "model__colsample_bytree": [0.8],
            }
            if settings.QUICK_TRAIN else
            {
                "model__n_estimators":    [20, 35, 50, 75, 100, 120, 135, 150],
                "model__max_depth":       [2, 4, 6, 8, 10],
                "model__learning_rate":   [0.03, 0.05, 0.1, 0.2],
                "model__subsample":       [0.6, 0.8, 1.0],
                "model__colsample_bytree":[0.6, 0.8, 1.0],
            }
        )
    }
}


# =============================================================================
# 🏋️  TRAIN — يُستدعى من main.py مع كل request
# =============================================================================
def train_on_df(df: pd.DataFrame) -> dict:
    """
    يأخذ DataFrame جاهز (بعد feature engineering) ويرجع bundle فيه:
        model, lags, roll, feature_columns, results
    """

    tscv    = TimeSeriesSplit(n_splits=5)
    results = []

    for LAGS in LAG_OPTIONS:
        for ROLL in ROLLING_OPTIONS:

            temp_df = df.copy()

            # ── Lag features ──────────────────────────────────────────────
            for lag in range(1, LAGS + 1):
                temp_df[f"lag_{lag}"] = (
                    temp_df.groupby("product_id")["number_of_product_purchases"]
                    .shift(lag)
                )

            # ── Rolling features ──────────────────────────────────────────
            temp_df[f"rolling_mean_{ROLL}"] = (
                temp_df.groupby("product_id")["number_of_product_purchases"]
                .rolling(ROLL).mean()
                .reset_index(0, drop=True)
            )
            temp_df[f"rolling_std_{ROLL}"] = (
                temp_df.groupby("product_id")["number_of_product_purchases"]
                .rolling(ROLL).std()
                .reset_index(0, drop=True)
            )

            temp_df = temp_df.dropna().reset_index(drop=True)

            x = temp_df.drop(columns=["number_of_product_purchases", "month"])
            y = temp_df["number_of_product_purchases"]

            split   = int(len(x) * 0.8)
            x_train = x.iloc[:split]
            x_val   = x.iloc[split:]
            y_train = y.iloc[:split]
            y_val   = y.iloc[split:]

            if len(y_val) <= 1:
                continue

            # ── Baseline RMSE (naive lag-1) ───────────────────────────────
            baseline_rmse = np.sqrt(mean_squared_error(
                y_val.iloc[1:], y_val.shift(1).iloc[1:]
            ))

            # ── Preprocessor ─────────────────────────────────────────────
            num_features = (
                [f"lag_{i}" for i in range(1, LAGS + 1)]
                + [f"rolling_mean_{ROLL}", f"rolling_std_{ROLL}",
                   "conversion_rate", "cart_drop_rate"]
            )

            numerical_pipeline = Pipeline([
                ("imputer", IterativeImputer(random_state=0)),
                ("scaler",  StandardScaler()),
            ])
            categorical_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            preprocessor = ColumnTransformer([
                ("num", numerical_pipeline,   num_features),
                ("cat", categorical_pipeline, ["product_id"]),
            ])

            # ── Grid search per model ─────────────────────────────────────
            for model_name, model_info in MODELS.items():

                pipe = Pipeline([
                    ("prep",  preprocessor),
                    ("model", model_info["model"]),
                ])

                gs = GridSearchCV(
                    pipe,
                    param_grid=model_info["params"],
                    cv=tscv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )
                gs.fit(x_train, y_train)

                best = gs.best_estimator_

                results.append({
                    "model":         model_name,
                    "lags":          LAGS,
                    "rolling":       ROLL,
                    "baseline_rmse": baseline_rmse,
                    "train_rmse":    np.sqrt(mean_squared_error(y_train, best.predict(x_train))),
                    "val_rmse":      np.sqrt(mean_squared_error(y_val,   best.predict(x_val))),
                    "best_params":   gs.best_params_,
                    "estimator":     best,
                    "feature_cols":  x_train.columns.tolist(),
                })

    # ── Pick best configuration ───────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df["composite_score"] = (
        results_df["val_rmse"]
        + 0.5 * (results_df["val_rmse"] - results_df["train_rmse"]).abs()
        + 0.2 * (results_df["val_rmse"] / results_df["baseline_rmse"])
    )

    best_row = results_df.sort_values("composite_score").iloc[0]

    return {
        "model":           best_row["estimator"],
        "lags":            int(best_row["lags"]),
        "roll":            int(best_row["rolling"]),
        "feature_columns": best_row["feature_cols"],
        "results": (
            results_df
            .drop(columns=["estimator", "feature_cols"])
            .to_dict(orient="records")
        ),
        "metrics": {
            "model_name":    best_row["model"],
            "baseline_rmse": best_row["baseline_rmse"],
            "train_rmse":    best_row["train_rmse"],
            "val_rmse":      best_row["val_rmse"],
        },
    }
#print(future_predictions_df)


# مجموع المبيعات لكل منتج
#product_sales = dff.groupby('product_id')['prediction'].sum().sort_values(ascending=False)

#top3_products = product_sales.head(3)
#print(top3_products)
#plt.figure(figsize=(8,5))
#plt.bar(top3_products.index, top3_products.values, color='green')
#plt.title('Top 3 Products by Total Purchases')
#plt.xlabel('Product ID')
#plt.ylabel('Total Purchases')



#bottom3_products = product_sales.tail(3)
#print(bottom3_products)
#plt.figure(figsize=(8,5))
#plt.bar(bottom3_products.index, bottom3_products.values, color='red')
#plt.title('Bottom 3 Products by Total Purchases')
#plt.xlabel('Product ID')
#plt.ylabel('Total Purchases')
#plt.show()