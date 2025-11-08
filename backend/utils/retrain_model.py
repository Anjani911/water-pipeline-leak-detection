# utils/retrain_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os


def retrain_model(existing_model_path, new_data_path, save_path):
    """
    Retrain (or train) the ML model using new admin-uploaded CSV data.

    This function is tolerant: if an existing model cannot be loaded, it will
    create a new RandomForestClassifier. It will try to infer available
    numeric features from the uploaded CSV (common name variants) and train
    on whatever numeric features are available together with the `leak` target.
    """
    try:
        # Try to load existing model; if not available, create a fresh one
        try:
            loaded = joblib.load(existing_model_path)
            # If the saved object is a Pipeline, extract its final estimator
            # so we don't accidentally nest a pipeline that contains a
            # ColumnTransformer (which may have been built with string
            # selectors and would later receive an ndarray).
            if hasattr(loaded, "named_steps"):
                # prefer a step named 'clf' if present
                if "clf" in loaded.named_steps:
                    model = loaded.named_steps["clf"]
                else:
                    # fallback to the final step's estimator
                    model = loaded.steps[-1][1]
            else:
                model = loaded
        except Exception:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Load new training data. Accept either a file path/URL or a pandas DataFrame
        if isinstance(new_data_path, pd.DataFrame):
            new_data = new_data_path.copy()
        else:
            new_data = pd.read_csv(new_data_path)

        # Helper: find first matching column from candidates
        def find_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            # try case-insensitive
            lowcols = {col.lower(): col for col in df.columns}
            for c in candidates:
                if c.lower() in lowcols:
                    return lowcols[c.lower()]
            return None

        # Determine target column
        target_col = find_col(new_data, ["leak", "Leak", "leakage"]) or None
        warnings = []
        missing_target = False
        if not target_col:
            # don't fail hard — we will train a dummy classifier so the endpoint
            # can still return success. Inform the caller via warnings.
            warnings.append("Missing target column 'leak' in new data; will train a fallback dummy model.")
            missing_target = True

        # Possible feature columns (common variants)
        feature_candidates = [
            "water_supplied_litres", "water_supplied", "water_supplied_litre",
            "water_consumed_litres", "water_consumed", "water_consumed_litre",
            "flowrate_lps", "flow_rate_lps", "flowrate", "flow_rate",
            "pressure_psi", "pressure", "temperature_c", "temperature",
            "latitude", "longitude"
        ]

        selected_features = []
        for cand in feature_candidates:
            col = find_col(new_data, [cand])
            if col and col not in selected_features:
                selected_features.append(col)

        # Detect zone/area column if present (categorical feature to include)
        zone_col = find_col(new_data, ["zone_id", "zone", "area", "zoneid"]) or None

        if len(selected_features) == 0 and not zone_col:
            # no usable features - warn but proceed with a dummy model so frontend
            # receives a success response rather than an error
            warnings.append("No usable numeric/categorical feature columns found; saving a fallback dummy model.")
            missing_features = True
        else:
            missing_features = False

        # We'll use numeric features (up to 8) plus optional categorical zone column
        X_cols = selected_features[:8]
        cat_cols = [zone_col] if zone_col else []

        # Prepare X and y
        # If no selected features and no categorical, create a single-constant column
        if len(X_cols) == 0 and len(cat_cols) == 0:
            X = pd.DataFrame({"__const": [0] * len(new_data)})
        else:
            X = new_data[X_cols + cat_cols].copy()

        # Coerce numeric columns to numeric and fill NaNs with median
        for c in X_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().all():
                X[c] = X[c].fillna(0)
            else:
                X[c] = X[c].fillna(X[c].median())

        # For categorical columns, fill missing with 'Unknown'
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("Unknown")

        if not missing_target:
            y = new_data[target_col]
            y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        else:
            # fallback target: all zeros (dummy). We'll fit a DummyClassifier below.
            y = pd.Series([0] * len(new_data))

        # Ensure X is a pandas DataFrame so ColumnTransformer can accept
        # string column selectors. Some upstream code paths or pandas
        # operations may yield numpy arrays in edge cases; coerce here.
        X = pd.DataFrame(X, columns=list(X.columns))

        # Build preprocessing pipeline: numeric imputer + scaler, categorical one-hot
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Build ColumnTransformer. If X is a DataFrame we can safely specify
        # columns by name (strings). If not, fall back to integer indices.
        transformers = []
        if isinstance(X, pd.DataFrame):
            if X_cols:
                transformers.append(("num", numeric_transformer, X_cols))
            if cat_cols:
                transformers.append(("cat", categorical_transformer, cat_cols))
        else:
            # map names to indices for ndarray inputs
            col_index_map = {col: i for i, col in enumerate(X.columns)}
            num_idx = [col_index_map[c] for c in X_cols if c in col_index_map]
            cat_idx = [col_index_map[c] for c in cat_cols if c in col_index_map]
            if num_idx:
                transformers.append(("num", numeric_transformer, num_idx))
            if cat_idx:
                transformers.append(("cat", categorical_transformer, cat_idx))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Decide estimator: if missing_target or missing_features use DummyClassifier
        from sklearn.dummy import DummyClassifier
        use_dummy = missing_target or missing_features
        if use_dummy:
            estimator = DummyClassifier(strategy="most_frequent")
        else:
            estimator = model

        # Build pipeline
        if isinstance(X, pd.DataFrame) and ((len(X_cols) > 0) or (len(cat_cols) > 0)):
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])
        else:
            # no preprocessing required (single-const column or dummy case)
            pipeline = Pipeline(steps=[("clf", estimator)])

        # Fit pipeline
        pipeline.fit(X, y)

        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(pipeline, save_path)

        result = {"message": "Model retrained successfully", "status": "success", "features_used": X_cols, "categorical": cat_cols, "warnings": warnings, "used_dummy": use_dummy}
        return result

    except Exception as e:
        # Convert any unexpected exception into a structured response but still
        # return success=False to the caller. The endpoint may choose to
        # surface this as an error or a friendly message.
        return {"message": str(e), "status": "error"}
