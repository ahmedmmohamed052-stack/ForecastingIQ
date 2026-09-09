"""
🧬  Automatic schema detection.

The original pipeline only accepted one fixed shape of CSV (month,
product_id, number_of_product_purchases, ...). This module lets /train and
/forecast accept ANY CSV with ANY column names by figuring out, on its own:

  - date_col    : which column is the time axis
  - group_col   : which column (if any) splits the data into separate series
                  to forecast independently — e.g. product_id, store_id,
                  customer_id. None means "treat the whole file as one series".
  - target_col  : which numeric column to forecast
  - numeric_features / categorical_features : everything else, fed to the
                  model as extra inputs (replaces the old hardcoded
                  conversion_rate / cart_drop_rate engineering, which only
                  made sense for the original e-commerce schema)

Detection runs once, at /train time, and the resulting dict is saved in the
model bundle (see main.py `_save_trained_model`) so /forecast reuses the
EXACT same columns instead of re-guessing on a new upload — the new CSV
must have the same column names as the one used for training.

Optional overrides (date_column / group_column / target_column) let a
caller pin a column explicitly instead of relying on auto-detection —
main.py exposes these as optional query params on /train.
"""
import pandas as pd

DATE_NAME_HINTS = ["date", "month", "day", "week", "year", "period", "ds", "timestamp", "time"]
TARGET_NAME_HINTS = [
    "target", "sales", "purchase", "purchases", "revenue", "demand", "qty", "quantity",
    "units", "amount", "count", "orders", "value", "y",
]
# Columns whose name suggests they're an intermediate/derived signal (a
# rate, a sub-count of some larger event) rather than the thing you'd
# actually want to forecast — used to break ties when several columns
# match a TARGET_NAME_HINT (e.g. a CSV with both "number_of_product_
# purchases" and "number_of_times_add_followed_by_purchase").
TARGET_NAME_PENALTIES = ["rate", "followed_by", "drop", "ratio", "pct", "percent"]
GROUP_NAME_HINTS = ["id", "product", "sku", "store", "customer", "category", "region", "segment", "item", "group"]

# Columns with more unique values than this (as a fraction of row count)
# are treated as free text / identifiers, not useful categorical features.
MAX_CATEGORY_UNIQUE_RATIO = 0.2
MAX_CATEGORY_UNIQUE_ABS = 50


def _date_parse_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return pd.to_datetime(series, errors="coerce").notna().mean()


def _numeric_parse_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return pd.to_numeric(series, errors="coerce").notna().mean()


def detect_date_column(df: pd.DataFrame, override: str = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"date_column '{override}' not found in the CSV. Columns: {list(df.columns)}")
        return override

    candidates = []
    for col in df.columns:
        ratio = _date_parse_ratio(df[col])
        if ratio >= 0.9:
            bonus = 0.05 if any(h in str(col).lower() for h in DATE_NAME_HINTS) else 0.0
            candidates.append((ratio + bonus, col))

    if not candidates:
        raise ValueError(
            "Couldn't find a date/time column in the CSV — at least one "
            "column needs to contain parseable dates (e.g. '2024-01', "
            "'Jan 2024', '2024-01-15')."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def detect_group_column(df: pd.DataFrame, date_col: str, override: str = None) -> str | None:
    if override:
        if override not in df.columns:
            raise ValueError(f"group_column '{override}' not found in the CSV. Columns: {list(df.columns)}")
        return override

    n = len(df)
    best, best_score = None, -1.0
    for col in df.columns:
        if col == date_col:
            continue
        nunique = df[col].nunique(dropna=True)
        # A usable "group" column has more than one row per value (so there's
        # a real time series per group) but isn't a near-constant column.
        if nunique <= 1 or nunique >= n:
            continue
        avg_rows_per_group = n / nunique
        if avg_rows_per_group < 2:
            continue
        score = avg_rows_per_group
        if any(h in str(col).lower() for h in GROUP_NAME_HINTS):
            score *= 1.5
        if score > best_score:
            best_score, best = score, col
    return best  # None => single series, no grouping


def detect_target_column(df: pd.DataFrame, date_col: str, group_col: str | None, override: str = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"target_column '{override}' not found in the CSV. Columns: {list(df.columns)}")
        return override

    reserved = {date_col, group_col} - {None}
    candidates = []
    for col in df.columns:
        if col in reserved:
            continue
        ratio = _numeric_parse_ratio(df[col])
        if ratio < 0.9:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.nunique(dropna=True) <= 1:
            continue  # constant — nothing to forecast
        # Name match strength: an exact hint match (column literally named
        # "sales") scores higher than a column that merely contains the
        # hint as a substring (e.g. "number_of_times_add_followed_by_
        # purchase" contains "purchase" but isn't really the target) —
        # this disambiguates CSVs with several "purchase-ish" columns.
        col_l = str(col).lower()
        name_score = 0.0
        for h in TARGET_NAME_HINTS:
            if col_l == h:
                name_score = max(name_score, 3.0)
            elif col_l.startswith(h + "_") or col_l.endswith("_" + h):
                name_score = max(name_score, 2.0)
            elif h in col_l:
                name_score = max(name_score, 1.0)
        if any(p in col_l for p in TARGET_NAME_PENALTIES):
            name_score -= 1.5
        variance = numeric.var()
        candidates.append((name_score, variance if pd.notna(variance) else 0.0, col))

    if not candidates:
        raise ValueError(
            "Couldn't find a numeric column to forecast in the CSV — need "
            "at least one numeric column besides the date/group columns."
        )
    # Name hints win first (exact > prefix/suffix > substring); highest
    # variance breaks ties among columns with the same name-match strength.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def detect_schema(
    df: pd.DataFrame,
    date_column: str = None,
    group_column: str = None,
    target_column: str = None,
) -> dict:
    """Returns {date_col, group_col, target_col, numeric_features, categorical_features}."""
    date_col = detect_date_column(df, override=date_column)
    group_col = detect_group_column(df, date_col, override=group_column)
    target_col = detect_target_column(df, date_col, group_col, override=target_column)

    reserved = {date_col, group_col, target_col} - {None}
    remaining = [c for c in df.columns if c not in reserved]

    numeric_features, categorical_features = [], []
    for col in remaining:
        if _numeric_parse_ratio(df[col]) >= 0.9:
            numeric_features.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            cap = max(MAX_CATEGORY_UNIQUE_ABS, len(df) * MAX_CATEGORY_UNIQUE_RATIO)
            if nunique < cap:
                categorical_features.append(col)
            # else: looks like free text / a unique identifier — dropped,
            # one-hot encoding it would blow up the model for no benefit.

    return {
        "date_col": date_col,
        "group_col": group_col,
        "target_col": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
