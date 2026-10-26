import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    # 1. Raise KeyError if any of the three required columns are missing
    for required_col in (target, id_col, time_col):
        if required_col not in df.columns:
            raise KeyError(f"Missing required column: {required_col}")

    n_rows = len(df)

    # Helper function: string that is empty or only whitespace counts as missing
    def is_missing_series(s: pd.Series) -> pd.Series:
        is_na = s.isna()
        # Convert string types to check for empty/whitespace-only values
        str_s = s.astype(str)
        is_blank = str_s.str.strip() == ""
        # Keep actual NA as NA (not converted to "nan" string false positive)
        return is_na | (is_blank & ~s.isna())

    # 2. missing: {column: fraction} for every column
    missing = {}
    for col in df.columns:
        if n_rows == 0:
            missing[col] = 0.0
        else:
            missing[col] = float(is_missing_series(df[col]).sum() / n_rows)

    # 3. duplicate_ids: sorted list of non-missing id_col values appearing > 1 time
    id_series = df[id_col]
    valid_ids = id_series[~is_missing_series(id_series)]
    duplicated_mask = valid_ids.duplicated(keep=False)
    duplicate_ids = sorted(valid_ids[duplicated_mask].unique().tolist())

    # 4. duplicate_rows: rows that exactly repeat an earlier row
    duplicate_rows = int(df.duplicated(keep="first").sum())

    # 5. constant_columns: sorted list of columns other than target with <= 1 distinct non-missing value
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        non_missing = df[col][~is_missing_series(df[col])]
        if non_missing.nunique() <= 1:
            constant_columns.append(col)
    constant_columns.sort()

    # 6. unlabelled_rows: count of rows where target is missing
    target_missing = is_missing_series(df[target])
    unlabelled_rows = int(target_missing.sum())

    # 7. positive_rate: share of labelled rows with target == 1 (0.0 if no row is labelled)
    labelled_df = df[~target_missing]
    if len(labelled_df) == 0:
        positive_rate = 0.0
    else:
        # Convert numeric string labels safely to integer/float comparison
        target_vals = pd.to_numeric(labelled_df[target], errors="coerce")
        positive_rate = float((target_vals == 1).sum() / len(labelled_df))

    # 8. bad_timestamps: non-missing values in time_col that do not parse as dates
    time_series = df[time_col]
    non_missing_time = time_series[~is_missing_series(time_series)]
    parsed_dates = pd.to_datetime(non_missing_time, errors="coerce")
    bad_timestamps = int(parsed_dates.isna().sum())

    # 9. time_range: [min, max] of parsed timestamps as ISO strings or [None, None]
    valid_dates = parsed_dates.dropna()
    if valid_dates.empty:
        time_range = [None, None]
    else:
        time_range = [valid_dates.min().isoformat(), valid_dates.max().isoformat()]

    # 10. leakage_suspects: evaluated on labelled rows when >= 20 labelled rows and both classes present
    leakage_suspects = []

    if len(labelled_df) >= 20 and labelled_df[target].nunique() == 2:
        y_labelled = pd.to_numeric(labelled_df[target], errors="coerce").astype(int)

        for col in df.columns:
            if col in (target, id_col, time_col):
                continue

            col_series = labelled_df[col]

            # Clarification rule: A column with a single distinct value is never a suspect (missing counts as a value)
            full_col_with_missing_tag = df[col].fillna("__MISSING_SENTINEL__").astype(str)
            if full_col_with_missing_tag.nunique() <= 1:
                continue

            is_suspect = False

            # Check (a): Numeric or boolean with >= 20 non-missing labelled values & both classes present
            non_missing_mask = ~is_missing_series(col_series)
            sub_col = col_series[non_missing_mask]
            sub_y = y_labelled[non_missing_mask]

            is_num_or_bool = (
                pd.api.types.is_numeric_dtype(col_series) or 
                pd.api.types.is_bool_dtype(col_series)
            )

            if is_num_or_bool and len(sub_col) >= 20 and sub_y.nunique() == 2:
                try:
                    auc_a = roc_auc_score(sub_y, sub_col.astype(float))
                    if auc_a >= 0.98 or auc_a <= 0.02:
                        is_suspect = True
                except Exception:
                    pass

            # Check (b): Missingness indicator ROC AUC against label on labelled rows
            if not is_suspect:
                missing_indicator = is_missing_series(col_series).astype(int)
                if missing_indicator.nunique() == 2:
                    try:
                        auc_b = roc_auc_score(y_labelled, missing_indicator)
                        if auc_b >= 0.98 or auc_b <= 0.02:
                            is_suspect = True
                    except Exception:
                        pass

            if is_suspect:
                leakage_suspects.append(col)

    leakage_suspects.sort()

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }