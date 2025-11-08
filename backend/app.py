# ============================================================
# 💧 WATER LEAKAGE DETECTION BACKEND (NO DATABASE VERSION)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import joblib
import pandas as pd
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from utils.blockchain import SimplePrivateBlockchain
from utils.retrain_model import retrain_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ------------------------------------------------------------
# 🔧 Basic Setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
LOG_FILE = "server_log.json"
USER_FILE = "users.json"
UPLOAD_RECORDS = "uploads.json"
LEDGER_FILE = "ledger.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "leak_detection_model.pkl")

try:
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model file not found at {MODEL_PATH} (will run without model).")
        model = None
    else:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print("⚠️ Error loading model:", e)
    model = None

blockchain = SimplePrivateBlockchain()

# ------------------------------------------------------------
# 🔹 Helper Utilities
# ------------------------------------------------------------
def load_json(path, default):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4)
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def log_event(action, username="system", details=None):
    logs = load_json(LOG_FILE, [])
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "username": username,
        "details": details or {}
    })
    save_json(LOG_FILE, logs)

# ------------------------------------------------------------
# 🔹 AUTH ENDPOINTS
# ------------------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "citizen")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    users = load_json(USER_FILE, [])
    if any(u["username"] == username for u in users):
        return jsonify({"error": "Username already exists"}), 409

    hashed = generate_password_hash(password)
    users.append({"username": username, "password": hashed, "role": role, "coins": 0, "reports": 0})
    save_json(USER_FILE, users)
    log_event("signup", username, {"role": role})
    return jsonify({"message": "✅ Signup successful"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    users = load_json(USER_FILE, [])
    for u in users:
        if u["username"] == username and check_password_hash(u["password"], password):
            log_event("login", username)
            return jsonify({"message": "✅ Login successful", "role": u["role"]}), 200
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "💧 Water Leakage Detection Backend is Running"}), 200

# ------------------------------------------------------------
# 🔹 ADMIN ENDPOINTS
# ------------------------------------------------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    uploads = load_json(UPLOAD_RECORDS, [])
    uploads.append({
        "username": "admin",
        "filename": filename,
        "filetype": "dataset",
        "path": filepath,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(UPLOAD_RECORDS, uploads)

    log_event("upload_dataset", "admin", {"filename": filename})
    return jsonify({"message": "✅ Dataset uploaded successfully", "path": filepath})


@app.route("/retrain_model", methods=["POST"])
def retrain_model_endpoint():
    try:
        # Accept uploaded CSV directly (no path required)
        if "file" not in request.files:
            return jsonify({"error": "No file provided. Please upload CSV with key 'file'."}), 400

        file = request.files["file"]
        # Read CSV into DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {e}"}), 400

        # Call retrain_model passing DataFrame directly; save model to MODEL_PATH
        result = retrain_model(MODEL_PATH, df, MODEL_PATH)
        if isinstance(result, dict) and result.get("status") == "success":
            log_event("retrain_model", "admin", {"rows": len(df)})
            return jsonify({"message": "✅ Model retrained successfully", "detail": result.get("message"), "features_used": result.get("features_used"), "categorical": result.get("categorical")}), 200
        else:
            msg = result.get("message") if isinstance(result, dict) else str(result)
            return jsonify({"error": msg}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/view_reports", methods=["GET"])
def view_reports():
    # Dummy data simulation
    report_data = {
        "summary": {"total_leaks": 12, "high_risk_areas": 5, "avg_confidence": 0.83},
        "graph_data": [
            {"area": "Sector 1", "confidence": 0.9},
            {"area": "Sector 2", "confidence": 0.85},
            {"area": "Sector 3", "confidence": 0.6},
        ],
        "table_data": [
            {"id": 1, "location": "Sector 1", "status": "Leak Detected"},
            {"id": 2, "location": "Sector 2", "status": "Possible Leak"},
        ]
    }
    return jsonify(report_data)


@app.route("/ledger", methods=["GET"])
def view_ledger():
    ledger = load_json(LEDGER_FILE, [])
    return jsonify({"ledger": ledger})


@app.route("/add_transaction", methods=["POST"])
def add_transaction():
    data = request.get_json()
    sender = data.get("sender")
    receiver = data.get("receiver")
    amount = data.get("amount")

    if not all([sender, receiver, amount]):
        return jsonify({"error": "Missing fields"}), 400

    new_block = blockchain.add_transaction(sender, receiver, amount)
    ledger = load_json(LEDGER_FILE, [])
    ledger.append(new_block)
    save_json(LEDGER_FILE, ledger)

    log_event("add_transaction", "admin", {"sender": sender, "receiver": receiver, "amount": amount})
    return jsonify({"message": "✅ Transaction added", "block": new_block})


@app.route("/logs", methods=["GET"])
def get_logs():
    logs = load_json(LOG_FILE, [])
    return jsonify({"logs": logs})

# ------------------------------------------------------------
# 🔹 CITIZEN ENDPOINTS
# ------------------------------------------------------------
@app.route("/citizen_profile", methods=["GET"])
def citizen_profile():
    username = request.args.get("username")
    users = load_json(USER_FILE, [])
    user = next((u for u in users if u["username"] == username), None)
    if not user:
        return jsonify({"error": "User not found"}), 404

    profile = {
        "username": user["username"],
        "coins": user["coins"],
        "reports": user["reports"],
        "city": "Raipur",
        "colony": "Shanti Nagar"
    }
    return jsonify(profile)


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    username = request.form.get("username", "anonymous")
    if "photo" not in request.files:
        return jsonify({"error": "No photo provided"}), 400

    photo = request.files["photo"]
    filename = secure_filename(photo.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    photo.save(filepath)

    uploads = load_json(UPLOAD_RECORDS, [])
    uploads.append({
        "username": username,
        "filename": filename,
        "filetype": "photo",
        "path": filepath,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(UPLOAD_RECORDS, uploads)

    users = load_json(USER_FILE, [])
    for u in users:
        if u["username"] == username:
            u["reports"] += 1
            u["coins"] += 5
    save_json(USER_FILE, users)

    log_event("upload_photo", username, {"filename": filename})
    return jsonify({"message": "📷 Photo uploaded successfully", "path": filepath})


@app.route("/my_reports", methods=["GET"])
def my_reports():
    username = request.args.get("username")
    uploads = load_json(UPLOAD_RECORDS, [])
    user_reports = [u for u in uploads if u["username"] == username and u["filetype"] == "photo"]
    return jsonify({"reports": user_reports})


@app.route("/my_rewards", methods=["GET"])
def my_rewards():
    username = request.args.get("username")
    users = load_json(USER_FILE, [])
    user = next((u for u in users if u["username"] == username), None)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"username": username, "coins": user["coins"]})

# ------------------------------------------------------------
# 🔹 ML / PREDICTION ENDPOINTS
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        df = pd.read_csv(file)

        if not model:
            return jsonify({"error": "Model not loaded"}), 500

        # Helper to normalize column names for matching
        def norm_col(c: str) -> str:
            return str(c).lower().strip().replace(" ", "_").replace("-", "_").replace("__", "_")

        # Build mapping of normalized incoming column -> original column name
        incoming = {norm_col(c): c for c in df.columns}

        # Determine expected features from the model if available
        expected_features = None
        try:
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
        except Exception:
            expected_features = None

        # Fallback expected features (common ones used in training)
        if not expected_features:
            expected_features = [
                "flow_rate_lps",
                "latitude",
                "longitude",
                "temperature_c",
            ]

        # Try to map incoming columns to expected feature names
        rename_map = {}
        for feat in expected_features:
            nfeat = norm_col(feat)
            if nfeat in incoming:
                rename_map[incoming[nfeat]] = feat
            else:
                # try variants: remove underscores
                compact = nfeat.replace("_", "")
                matched = next((orig for k, orig in incoming.items() if k.replace("_", "") == compact), None)
                if matched:
                    rename_map[matched] = feat

        # Apply renaming so DataFrame columns match expected feature names where possible
        if rename_map:
            df = df.rename(columns=rename_map)

        # For any expected feature missing, add with a safe default (0 or median if possible)
        missing_features = [f for f in expected_features if f not in df.columns]
        for mf in missing_features:
            # prefer numeric median from other columns if available, else 0
            default_val = 0
            try:
                # if DataFrame has numeric columns, use their median as a reasonable default
                if not df.select_dtypes(include=["number"]).empty:
                    default_val = float(df.select_dtypes(include=["number"]).median().median())
            except Exception:
                default_val = 0
            df[mf] = default_val

        # Prepare input matrix: select expected features in order
        X = df[expected_features].copy()

        # Coerce expected feature columns to numeric where possible. Non-numeric values
        # (like 'Devendra Nagar Zone') will become NaN and then be filled with a
        # sensible default (column median or 0). This avoids conversion errors.
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if X[col].isna().all():
                    # if entire column became NaN, fill with 0
                    X[col] = X[col].fillna(0)
                else:
                    # fill NaNs with median of that column
                    med = float(X[col].median()) if not X[col].dropna().empty else 0.0
                    X[col] = X[col].fillna(med)
            except Exception:
                # safest fallback: replace non-numeric with 0
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # Attempt prediction. Prefer passing the full DataFrame to let a saved
        # pipeline (preprocessor + estimator) handle column selection/encoding.
        preds = None
        try:
            preds = model.predict(df)
        except Exception:
            # If that fails, try the prepared numeric matrix X (older models)
            try:
                preds = model.predict(X)
            except Exception:
                try:
                    preds = model.predict(X.values)
                except Exception as e2:
                    traceback.print_exc()
                    return jsonify({"error": "Prediction failed: " + str(e2)}), 500

        # Normalize predictions to ints where possible
        try:
            preds_list = [int(x) for x in (preds.tolist() if hasattr(preds, "tolist") else preds)]
        except Exception:
            preds_list = list(preds)

        df["Predicted_Leak"] = preds_list

        leak_count = int(sum(1 for v in preds_list if v == 1))
        no_leak_count = int(sum(1 for v in preds_list if v == 0))

        # Compute per-zone summary if zone column exists
        zone_candidates = ["zone_id", "zone", "area", "zoneid"]
        zone_col = None
        for c in df.columns:
            if c.lower() in zone_candidates:
                zone_col = c
                break
        zone_summary = []
        if zone_col:
            grouped = df.groupby(zone_col)["Predicted_Leak"].agg(["sum", "count"]).reset_index()
            for _, row in grouped.iterrows():
                zone_summary.append({
                    "zone": row[zone_col],
                    "leak_count": int(row["sum"]),
                    "total": int(row["count"]),
                    "leak_rate": float(row["sum"]) / int(row["count"]) if int(row["count"])>0 else 0.0
                })

        result = df.fillna("").to_dict(orient="records")
        log_event("predict", "admin", {"rows": len(result), "leak_count": leak_count})

        return jsonify({
            "summary": {"leak_count": leak_count, "no_leak_count": no_leak_count},
            "zone_summary": zone_summary,
            "data": result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/report_leak", methods=["POST"])
def report_leak():
    try:
        data = request.get_json()
        leaks = data.get("leaks", [])
        log_event("report_leak", "system", {"count": len(leaks)})
        return jsonify({"message": f"✅ {len(leaks)} leaks reported successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/heatmap_by_zone", methods=["POST"])
def heatmap_by_zone():
    """Accepts a CSV upload and returns per-zone centroid points with max flow and a
    normalized weight suitable for constructing a heatmap on the frontend.

    Request: multipart/form-data with key 'file' -> CSV
             optional form fields: 'grid_size' (ignored here, kept for parity),
             'flow_candidates' (comma-separated column names to prefer)
    Response: JSON { points: [[lat, lon, weight, zone, max_flow], ...], bounds: [minLat, minLon, maxLat, maxLon], zone_col, flow_col }
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided. Please upload CSV with key 'file'."}), 400

        file = request.files["file"]
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {e}"}), 400

        def norm_col(c: str) -> str:
            return str(c).lower().strip().replace(" ", "_").replace("-", "_").replace("__", "_")

        incoming = {norm_col(c): c for c in df.columns}

        # identify zone column
        zone_candidates = ["zone_id", "zone", "area", "zoneid", "zone_name"]
        zone_col = None
        for k, orig in incoming.items():
            if k in zone_candidates:
                zone_col = orig
                break
        if not zone_col:
            # try fuzzy match by substring
            for k, orig in incoming.items():
                if any(x in k for x in ["zone", "area"]):
                    zone_col = orig
                    break

        if not zone_col:
            return jsonify({"error": "No zone/area column found in CSV. Expected one of: zone_id, zone, area, zoneid"}), 400

        # identify latitude / longitude columns (optional but recommended)
        lat_candidates = ["latitude", "lat", "y", "lat_dd"]
        lon_candidates = ["longitude", "lon", "lng", "long", "long_dd"]
        lat_col = None
        lon_col = None
        for k, orig in incoming.items():
            if not lat_col and k in lat_candidates:
                lat_col = orig
            if not lon_col and k in lon_candidates:
                lon_col = orig

        # identify best flow column from common candidates or user-supplied list
        default_flow_candidates = [
            "flowrate_lps", "flow_rate_lps", "flowrate", "flow_rate",
            "water_supplied_litres", "water_supplied", "water_consumed_litres", "water_consumed",
            "flow_lps", "max_flow"
        ]
        # allow override from form
        fc = request.form.get("flow_candidates")
        if fc:
            user_cands = [norm_col(x) for x in fc.split(",") if x.strip()]
            # prepend user-specified to default search order
            search_candidates = user_cands + default_flow_candidates
        else:
            search_candidates = default_flow_candidates

        flow_col = None
        for cand in search_candidates:
            if cand in incoming:
                flow_col = incoming[cand]
                break
            # also try compact match
            compact = cand.replace("_", "")
            matched = next((orig for k, orig in incoming.items() if k.replace("_", "") == compact), None)
            if matched:
                flow_col = matched
                break

        if not flow_col:
            return jsonify({"error": "No flow-related column found in CSV. Please include a column like 'flowrate_lps' or 'water_supplied_litres', or pass flow_candidates form field."}), 400

        # Group by zone and compute max flow and centroid (if lat/lon available)
        results = []
        grouped = df.groupby(zone_col)
        for zone_name, grp in grouped:
            # coerce flow to numeric
            try:
                vals = pd.to_numeric(grp[flow_col], errors="coerce").dropna()
                max_flow = float(vals.max()) if not vals.empty else 0.0
            except Exception:
                max_flow = 0.0

            lat_val = None
            lon_val = None
            if lat_col and lon_col and lat_col in grp.columns and lon_col in grp.columns:
                try:
                    lat_val = float(pd.to_numeric(grp[lat_col], errors="coerce").dropna().mean())
                    lon_val = float(pd.to_numeric(grp[lon_col], errors="coerce").dropna().mean())
                except Exception:
                    lat_val = None
                    lon_val = None

            results.append({"zone": zone_name, "max_flow": max_flow, "lat": lat_val, "lon": lon_val})

        # Filter out zones without coordinates for heatmap points; still return their numbers
        points = []
        flows = [r["max_flow"] for r in results]
        if not flows:
            return jsonify({"error": "No zones found in CSV."}), 400

        fmin = min(flows)
        fmax = max(flows)
        for r in results:
            if r["lat"] is None or r["lon"] is None:
                # skip spatial-less zones for heatmap plotting
                continue
            if fmax > fmin:
                weight = (r["max_flow"] - fmin) / (fmax - fmin)
            else:
                weight = 1.0
            points.append([r["lat"], r["lon"], weight, r["zone"], r["max_flow"]])

        if not points:
            # no spatial points available; return per-zone stats so frontend can decide
            return jsonify({"points": [], "zones": results, "message": "No latitude/longitude present for zones. Zones provided with max_flow values."}), 200

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        bounds = [min(lats), min(lons), max(lats), max(lons)]

        return jsonify({"points": points, "bounds": bounds, "zone_col": zone_col, "flow_col": flow_col})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    print("✅ Flask backend running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
