"""
FairLens v2 — Universal Bias Detection Backend
Works on ANY user-uploaded CSV dataset.
No preloaded demo data dependency.

Run:
  pip install flask flask-cors pandas numpy scikit-learn
  python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, sqlite3, os, traceback, io
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Optional Fairlearn
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_OK = True
except ImportError:
    FAIRLEARN_OK = False

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), "fairlens.db")

# ─── Keywords for auto-detection ─────────────────────────────────────────────
SENSITIVE_KW = {
    "gender": ["gender", "sex", "male", "female", "genderidentity", "gender_identity"],
    "race":   ["race", "ethnicity", "ethnic", "race_ethnicity", "nationality"],
    "age":    ["age", "age_group", "agegroup", "age_bracket"],
}
TARGET_KW = [
    "approved", "approval", "hired", "outcome", "label", "target",
    "decision", "result", "granted", "rejected", "selected", "passed",
    "loan_approved", "credit_approved", "y", "class", "flag"
]

# ─── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, timestamp TEXT,
        sensitive_col TEXT, target_col TEXT,
        before_json TEXT, after_json TEXT,
        improvement_json TEXT, n_rows INTEGER
    )""")
    conn.commit(); conn.close()

def save_to_db(filename, sensitive_col, target_col, before, after, improvement, n_rows):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO analyses
        (filename,timestamp,sensitive_col,target_col,before_json,after_json,improvement_json,n_rows)
        VALUES (?,?,?,?,?,?,?,?)""",
        (filename, datetime.now().isoformat(), sensitive_col, target_col,
         json.dumps(before), json.dumps(after), json.dumps(improvement), n_rows))
    conn.commit(); rid = c.lastrowid; conn.close()
    return rid

def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,filename,timestamp,sensitive_col,target_col,n_rows FROM analyses ORDER BY id DESC LIMIT 20")
    rows = c.fetchall(); conn.close()
    return [{"id":r[0],"filename":r[1],"timestamp":r[2],"sensitive_col":r[3],"target_col":r[4],"n_rows":r[5]} for r in rows]

def load_by_id(aid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id=?", (aid,))
    r = c.fetchone(); conn.close()
    if not r: return None
    return {"id":r[0],"filename":r[1],"timestamp":r[2],
            "before":json.loads(r[5]),"after":json.loads(r[6]),"improvement":json.loads(r[7])}

# ─── Data Cleaning ────────────────────────────────────────────────────────────
def clean_dataframe(df):
    """Universal cleaning: handle missing values, strip whitespace."""
    df = df.copy()
    # Strip column name whitespace
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Drop rows where more than 50% values missing
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    # Fill numeric NaN with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    # Fill categorical NaN with mode
    for col in df.select_dtypes(include=["object"]).columns:
        mode = df[col].mode()
        if len(mode) > 0:
            df[col] = df[col].fillna(mode[0])
        df[col] = df[col].str.strip()
    return df

# ─── Column Auto-Detection ────────────────────────────────────────────────────
def detect_columns(df):
    """
    Returns:
      sensitive_attrs: dict {attr_type: col_name}
      target_col: str or None
      binary_cols: list of all binary columns
    """
    cols_lower = {c.lower(): c for c in df.columns}
    sensitive_attrs = {}

    for attr, keywords in SENSITIVE_KW.items():
        for kw in keywords:
            if kw in cols_lower:
                sensitive_attrs[attr] = cols_lower[kw]
                break
            # Partial match
            for col_l, col_orig in cols_lower.items():
                if kw in col_l and attr not in sensitive_attrs:
                    sensitive_attrs[attr] = col_orig

    # Detect target: keyword match first
    target_col = None
    for kw in TARGET_KW:
        if kw in cols_lower:
            candidate = cols_lower[kw]
            uniq = df[candidate].dropna().unique()
            if len(uniq) <= 6:  # loose binary check
                target_col = candidate
                break

    # Fallback: find any binary column (0/1 or Yes/No)
    binary_cols = []
    for col in df.columns:
        uniq = set(str(v).lower().strip() for v in df[col].dropna().unique())
        if uniq <= {"0","1","yes","no","true","false","1.0","0.0","y","n"} and len(uniq) >= 2:
            binary_cols.append(col)
            if target_col is None and col not in sensitive_attrs.values():
                target_col = col

    return sensitive_attrs, target_col, binary_cols

def binarize_target(series):
    """Convert Yes/No/True/False → 1/0."""
    mapping = {"yes":1,"no":0,"true":1,"false":0,"y":1,"n":0,"1":1,"0":0,"1.0":1,"0.0":0}
    def conv(v):
        s = str(v).lower().strip()
        return mapping.get(s, int(float(s)) if s.replace('.','').isdigit() else np.nan)
    return series.apply(conv)

# ─── Fairness Metrics ─────────────────────────────────────────────────────────
def compute_metrics(df, sensitive_col, target_col, predictions):
    predictions = np.array(predictions)
    groups = df[sensitive_col].astype(str).unique()
    group_stats = {}

    for g in groups:
        mask = (df[sensitive_col].astype(str) == g).values
        n = mask.sum()
        if n == 0: continue
        approved = int(predictions[mask].sum())
        rate = approved / n
        actual = int(df.loc[mask, target_col].sum())
        actual_rate = actual / n
        # TPR
        pos_mask = mask & (df[target_col].values == 1)
        tp = int((predictions[pos_mask] == 1).sum())
        tpr = tp / pos_mask.sum() if pos_mask.sum() > 0 else 0.0
        group_stats[g] = {
            "n": int(n),
            "approved": approved,
            "approval_rate": round(float(rate), 4),
            "actual_approval_rate": round(float(actual_rate), 4),
            "tpr": round(float(tpr), 4)
        }

    rates = [s["approval_rate"] for s in group_stats.values() if s["n"] > 0]
    if not rates:
        return {}
    max_r, min_r = max(rates), min(rates)

    spd = round(max_r - min_r, 4)
    di  = round(min_r / max_r, 4) if max_r > 0 else 1.0
    acc = round(float(np.mean(predictions == df[target_col].values)), 4)

    # Equal opportunity
    tprs = [s["tpr"] for s in group_stats.values()]
    eo_diff = round(max(tprs) - min(tprs), 4) if tprs else 0.0

    # Fairness score (0–100)
    di_score  = min(di / 0.8, 1.0) * 35
    spd_score = max(0, 1 - spd / 0.3) * 35
    eo_score  = max(0, 1 - eo_diff / 0.3) * 30
    fs = round(di_score + spd_score + eo_score, 1)

    bias_level = "HIGH" if fs < 50 else ("MEDIUM" if fs < 75 else "LOW")
    bias_color = "red"  if fs < 50 else ("yellow" if fs < 75 else "green")
    bias_detected = spd > 0.10 or di < 0.80

    # Confidence score: more data = higher confidence
    n_total = sum(s["n"] for s in group_stats.values())
    confidence = min(100, round(50 + (n_total / 20), 1))

    return {
        "accuracy": acc,
        "statistical_parity_difference": spd,
        "disparate_impact": di,
        "equal_opportunity_difference": eo_diff,
        "fairness_score": fs,
        "bias_level": bias_level,
        "bias_color": bias_color,
        "bias_detected": bias_detected,
        "confidence_score": confidence,
        "group_stats": group_stats,
        "privileged_group": max(group_stats, key=lambda g: group_stats[g]["approval_rate"]),
        "disadvantaged_group": min(group_stats, key=lambda g: group_stats[g]["approval_rate"]),
        "n_total": n_total
    }

# ─── Bias Explanation ─────────────────────────────────────────────────────────
def build_explanation(df, sensitive_col, target_col, m_before, m_after=None):
    gs = m_before["group_stats"]
    priv = m_before["privileged_group"]
    disadv = m_before["disadvantaged_group"]
    pr = gs[priv]["approval_rate"] * 100
    dr = gs[disadv]["approval_rate"] * 100
    gap = pr - dr
    total = m_before["n_total"]
    di = m_before["disparate_impact"]
    spd = m_before["statistical_parity_difference"]

    exps = []

    # Main explanation
    main = (f"Bias detected because '{disadv}' group has a {gap:.1f}% lower approval rate "
            f"({dr:.1f}%) than '{priv}' ({pr:.1f}%). "
            f"Disparate Impact = {di:.2f} (threshold: 0.80).")
    exps.append({"type": "Root Cause", "icon": "🔍", "text": main,
                 "severity": "HIGH" if di < 0.6 else "MEDIUM"})

    # Data size
    disadv_n = gs[disadv]["n"]
    priv_n = gs[priv]["n"]
    if disadv_n < priv_n * 0.6:
        exps.append({"type": "Underrepresentation", "icon": "📊",
                     "text": f"'{disadv}' has only {disadv_n} samples vs {priv_n} for '{priv}'. "
                             f"Underrepresentation causes the model to learn biased patterns.",
                     "severity": "HIGH"})

    # Legal threshold
    if di < 0.8:
        exps.append({"type": "Legal Violation Risk", "icon": "⚠️",
                     "text": f"Disparate Impact {di:.2f} is below the 0.80 legal threshold (EEOC 4/5ths Rule). "
                             f"This may constitute adverse impact discrimination.",
                     "severity": "HIGH" if di < 0.6 else "MEDIUM"})

    # SPD
    if spd > 0.15:
        exps.append({"type": "Statistical Parity Failure", "icon": "📉",
                     "text": f"Statistical Parity Difference = {spd:.2f}. "
                             f"The model assigns different approval standards to different groups. Ideal value: 0.",
                     "severity": "MEDIUM"})

    # After fix
    if m_after:
        imp = m_after["fairness_score"] - m_before["fairness_score"]
        exps.append({"type": "Mitigation Result", "icon": "✅",
                     "text": f"After applying bias mitigation, fairness score improved by +{imp:.1f} points. "
                             f"Disparate Impact: {m_before['disparate_impact']:.2f} → {m_after['disparate_impact']:.2f}.",
                     "severity": "IMPROVEMENT"})

    return exps

# ─── Recommendations ──────────────────────────────────────────────────────────
def build_recommendations(m):
    recs = []
    di, spd, eo = m["disparate_impact"], m["statistical_parity_difference"], m["equal_opportunity_difference"]

    # Best fix recommendation
    if di < 0.7:
        best = "Threshold Adjustment Recommended — fastest and most interpretable fix."
    elif spd > 0.2:
        best = "Reweighing Recommended — balances training data without model retraining."
    else:
        best = "Post-processing calibration recommended — minimal accuracy cost."
    recs.append({"priority":"HIGH","icon":"🎯","title":"Best Fix Method","detail": best})

    if di < 0.8:
        recs.append({"priority":"HIGH","icon":"🔴","title":"Collect Balanced Data",
                     "detail":"Actively collect more data from underrepresented groups to equalize training distribution."})
    if spd > 0.2:
        recs.append({"priority":"HIGH","icon":"🔴","title":"Remove Sensitive Attribute Influence",
                     "detail":"Use fairness constraints during training to prevent the sensitive attribute from driving predictions."})
    if eo > 0.15:
        recs.append({"priority":"MEDIUM","icon":"🟡","title":"Adjust Thresholds Per Group",
                     "detail":"Apply different decision thresholds for each group to equalize true positive rates."})
    recs.append({"priority":"LOW","icon":"🟢","title":"Deploy Ongoing Monitoring",
                 "detail":"Set automated alerts when fairness metrics degrade in production."})
    recs.append({"priority":"LOW","icon":"🟢","title":"Create Model Card",
                 "detail":"Document known biases, intended use, and fairness evaluations per EU AI Act requirements."})
    return recs

# ─── Bias Heatmap (feature contribution) ─────────────────────────────────────
def compute_feature_bias(df, sensitive_col, target_col, feature_cols):
    """Correlate each feature with the sensitive attribute — proxy bias measure."""
    result = []
    groups = df[sensitive_col].astype(str).unique()
    if len(groups) < 2:
        return result
    g1, g2 = groups[0], groups[1]
    m1 = df[df[sensitive_col].astype(str) == g1]
    m2 = df[df[sensitive_col].astype(str) == g2]

    for col in feature_cols[:10]:  # top 10
        try:
            v1 = m1[col].mean()
            v2 = m2[col].mean()
            overall = df[col].mean()
            if overall == 0: continue
            gap = abs(v1 - v2) / (abs(overall) + 1e-9)
            result.append({
                "feature": col,
                "group1_mean": round(float(v1), 3),
                "group2_mean": round(float(v2), 3),
                "gap_pct": round(float(gap * 100), 1),
                "bias_contribution": "HIGH" if gap > 0.3 else ("MEDIUM" if gap > 0.1 else "LOW")
            })
        except Exception:
            pass
    result.sort(key=lambda x: x["gap_pct"], reverse=True)
    return result[:8]

# ─── Mitigation Pipeline ──────────────────────────────────────────────────────
def encode_dataframe(df, target_col):
    """Encode all categoricals, return X, y, feature_cols, encoders."""
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.select_dtypes(include=["object"]).columns:
        if col == target_col: continue
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    feature_cols = [c for c in df_enc.columns if c != target_col]
    X = df_enc[feature_cols].fillna(0).values.astype(float)
    y = df_enc[target_col].fillna(0).values.astype(int)
    return X, y, feature_cols, encoders

def apply_reweighing(df, sensitive_col, target_col):
    n = len(df)
    weights = np.ones(n)
    for g in df[sensitive_col].unique():
        for o in df[target_col].unique():
            exp = (df[sensitive_col] == g).sum() / n * (df[target_col] == o).sum() / n
            obs = ((df[sensitive_col] == g) & (df[target_col] == o)).sum() / n
            w = exp / obs if obs > 0 else 1.0
            mask = ((df[sensitive_col] == g) & (df[target_col] == o)).values
            weights[mask] = w
    return weights

def apply_threshold_adjustment(df, sensitive_col, probas, overall_rate):
    """Find per-group threshold that brings each group's rate close to overall."""
    predictions = np.zeros(len(df), dtype=int)
    for g in df[sensitive_col].unique():
        mask = (df[sensitive_col] == g).values
        gp = probas[mask]
        best_t, best_diff = 0.5, float("inf")
        for t in np.arange(0.25, 0.85, 0.02):
            rate = (gp >= t).mean()
            diff = abs(rate - overall_rate)
            if diff < best_diff:
                best_diff = diff; best_t = t
        predictions[mask] = (gp >= best_t).astype(int)
    return predictions

def run_mitigation(df, sensitive_col, target_col, methods):
    X, y, feature_cols, _ = encode_dataframe(df, target_col)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Stage 1: Reweighing
    weights = None
    if "pre" in methods:
        weights = apply_reweighing(df, sensitive_col, target_col)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    if weights is not None:
        clf.fit(Xs, y, sample_weight=weights)
    else:
        clf.fit(Xs, y)

    probas = clf.predict_proba(Xs)[:, 1]

    # Stage 2: Fairlearn
    if "in" in methods and FAIRLEARN_OK:
        try:
            base = LogisticRegression(max_iter=1000, random_state=42)
            eg = ExponentiatedGradient(base, DemographicParity())
            eg.fit(Xs, y, sensitive_features=df[sensitive_col].astype(str).values)
            return eg.predict(Xs)
        except Exception as e:
            print(f"[WARN] Fairlearn failed: {e}")

    # Stage 3: Threshold adjustment
    overall_rate = y.mean()
    if "post" in methods:
        return apply_threshold_adjustment(df, sensitive_col, probas, overall_rate)

    return (probas >= 0.5).astype(int)

# ─── Baseline Model ───────────────────────────────────────────────────────────
def run_baseline(df, target_col):
    X, y, feature_cols, _ = encode_dataframe(df, target_col)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xs, y)
    preds = clf.predict(Xs)
    probas = clf.predict_proba(Xs)[:, 1]

    # Feature importances
    try:
        coef = np.abs(clf.coef_[0])
        fi = sorted(zip(feature_cols, coef.tolist()), key=lambda x: x[1], reverse=True)[:8]
    except:
        fi = []

    return preds, probas, feature_cols, fi

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","fairlearn":FAIRLEARN_OK,"ts":datetime.now().isoformat()})

# ── Step 1: Upload & Inspect CSV ──────────────────────────────────────────────
@app.route("/api/inspect", methods=["POST"])
def inspect():
    """
    Receive CSV, clean it, auto-detect columns.
    Returns column list, detected sensitive + target cols.
    Frontend can confirm or override via dropdowns.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file provided"}), 400
        f = request.files["file"]
        if not f.filename.endswith(".csv"):
            return jsonify({"error":"Only CSV files are supported"}), 400

        df_raw = pd.read_csv(f)
        if len(df_raw) < 5:
            return jsonify({"error":"Dataset too small (need at least 5 rows)"}), 400

        df = clean_dataframe(df_raw)
        sensitive_attrs, target_col, binary_cols = detect_columns(df)

        # Column metadata
        col_info = []
        for col in df.columns:
            uniq = df[col].dropna().unique()
            col_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_count": int(len(uniq)),
                "sample_values": [str(v) for v in uniq[:5]],
                "null_count": int(df[col].isna().sum()),
                "is_binary": col in binary_cols
            })

        # All columns as options for dropdowns
        all_cols = list(df.columns)

        return jsonify({
            "success": True,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": col_info,
            "all_columns": all_cols,
            "detected": {
                "sensitive_attrs": sensitive_attrs,
                "target_col": target_col,
                "binary_cols": binary_cols
            },
            "needs_manual": not sensitive_attrs or not target_col,
            "preview": df.head(5).to_dict(orient="records")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Step 2: Run Full Analysis ─────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full bias detection + optional mitigation.
    Accepts: CSV file + sensitive_col + target_col + mitigation flags
    """
    try:
        # Parse file
        if "file" not in request.files:
            return jsonify({"error":"No file provided"}), 400
        f = request.files["file"]
        filename = f.filename
        df_raw = pd.read_csv(f)
        df = clean_dataframe(df_raw)

        # Get column choices (user confirmed or auto-detected)
        sensitive_col = request.form.get("sensitive_col", "").strip()
        target_col    = request.form.get("target_col", "").strip()
        mitigation    = request.form.get("mitigation", "pre,post").split(",")

        # Validate columns exist
        if sensitive_col not in df.columns:
            return jsonify({"error": f"Column '{sensitive_col}' not found. Available: {list(df.columns)}"}), 400
        if target_col not in df.columns:
            return jsonify({"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}), 400

        # Binarize target
        df[target_col] = binarize_target(df[target_col])
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)

        # Drop rows where sensitive col is null
        df = df.dropna(subset=[sensitive_col])

        if len(df) < 10:
            return jsonify({"error":"Not enough clean data rows after processing (need ≥ 10)"}), 400

        # Validate: sensitive col has multiple groups
        groups = df[sensitive_col].astype(str).unique()
        if len(groups) < 2:
            return jsonify({"error":f"Column '{sensitive_col}' has only 1 unique value — cannot compare groups"}), 400

        # ── Baseline predictions ──────────────────────────────────────────────
        before_preds, before_probas, feature_cols, feature_importances = run_baseline(df, target_col)
        m_before = compute_metrics(df, sensitive_col, target_col, before_preds)

        if not m_before:
            return jsonify({"error":"Could not compute fairness metrics — check column values"}), 400

        # ── Mitigation (only if bias detected) ───────────────────────────────
        bias_found = m_before["bias_detected"]
        apply_mit = bias_found and any(m in mitigation for m in ["pre","in","post"])

        if apply_mit:
            after_preds = run_mitigation(df, sensitive_col, target_col, mitigation)
            m_after = compute_metrics(df, sensitive_col, target_col, after_preds)
        else:
            m_after = m_before.copy()

        # ── Improvement ───────────────────────────────────────────────────────
        improvement = {
            "fairness_score":     round(m_after["fairness_score"]     - m_before["fairness_score"], 1),
            "disparate_impact":   round(m_after["disparate_impact"]    - m_before["disparate_impact"], 4),
            "statistical_parity": round(m_before["statistical_parity_difference"] - m_after["statistical_parity_difference"], 4),
            "equal_opportunity":  round(m_before["equal_opportunity_difference"]  - m_after["equal_opportunity_difference"], 4),
            "pct_improvement":    round(
                (m_after["fairness_score"] - m_before["fairness_score"]) /
                max(100 - m_before["fairness_score"], 1) * 100, 1),
            "bias_fixed": bias_found and m_after["fairness_score"] > m_before["fairness_score"]
        }

        # ── XAI ───────────────────────────────────────────────────────────────
        explanations = build_explanation(df, sensitive_col, target_col, m_before,
                                         m_after if apply_mit else None)

        # ── Recommendations ───────────────────────────────────────────────────
        recommendations = build_recommendations(m_before)

        # ── Feature bias heatmap ──────────────────────────────────────────────
        numeric_features = [c for c in feature_cols
                            if c != sensitive_col and pd.api.types.is_numeric_dtype(df[c])]
        feature_bias = compute_feature_bias(df, sensitive_col, target_col, numeric_features)

        # ── Save to DB ────────────────────────────────────────────────────────
        aid = save_to_db(filename, sensitive_col, target_col, m_before, m_after,
                         improvement, len(df))

        return jsonify({
            "analysis_id": aid,
            "column_info": {
                "sensitive_col": sensitive_col,
                "target_col": target_col,
                "total_rows": len(df),
                "groups": list(groups),
                "features_used": feature_cols
            },
            "before": m_before,
            "after": m_after,
            "improvement": improvement,
            "bias_detected": bias_found,
            "apply_mit": apply_mit,
            "mitigation_methods": mitigation,
            "explanations": explanations,
            "recommendations": recommendations,
            "feature_importances": feature_importances,
            "feature_bias": feature_bias,
            "dataset_preview": df.head(6).to_dict(orient="records")
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── What-If Simulator ─────────────────────────────────────────────────────────
@app.route("/api/whatif", methods=["POST"])
def whatif():
    """
    Accepts a JSON row with field values. Returns prediction + counterfactual
    (flipping only the sensitive attribute).
    """
    try:
        body = request.get_json()
        inputs = body.get("inputs", {})
        sensitive_col = body.get("sensitive_col", "gender")
        model_info = body.get("model_info", {})  # coefs from last analysis

        # Simple logistic score from passed coefficients
        coefs = model_info.get("coefs", {})
        intercept = model_info.get("intercept", 0)

        def score_inputs(inp_dict):
            s = float(intercept)
            for k, v in inp_dict.items():
                if k in coefs:
                    try: s += float(coefs[k]) * float(v)
                    except: pass
            return 1 / (1 + np.exp(-s))

        prob = score_inputs(inputs)
        pred = 1 if prob >= 0.5 else 0

        # Counterfactual: flip sensitive attribute value
        cf_inputs = dict(inputs)
        orig_val = str(inputs.get(sensitive_col, ""))
        # Flip logic: if binary, flip; if "Male"/"Female", swap
        flip_map = {"male":"Female","female":"Male","0":"1","1":"0",
                    "yes":"No","no":"Yes","m":"F","f":"M","0.0":"1.0","1.0":"0.0"}
        cf_val = flip_map.get(orig_val.lower(), "Other")
        cf_inputs[sensitive_col] = cf_val

        cf_prob = score_inputs(cf_inputs)
        cf_pred = 1 if cf_prob >= 0.5 else 0
        changed = pred != cf_pred

        return jsonify({
            "prediction": pred,
            "probability": round(float(prob), 4),
            "outcome": "Approved ✓" if pred else "Denied ✗",
            "counterfactual": {
                "changed_attr": sensitive_col,
                "original_value": orig_val,
                "flipped_value": cf_val,
                "prediction": cf_pred,
                "probability": round(float(cf_prob), 4),
                "outcome": "Approved ✓" if cf_pred else "Denied ✗",
                "decision_changed": changed,
                "alert": (
                    f"⚠️ Changing ONLY '{sensitive_col}' from '{orig_val}' → '{cf_val}' "
                    f"{'changes' if changed else 'does NOT change'} the decision. "
                    + ("This is direct evidence of bias." if changed else "Other factors dominate this case.")
                )
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── History ────────────────────────────────────────────────────────────────────
@app.route("/api/history")
def history():
    return jsonify(get_history())

@app.route("/api/history/<int:aid>")
def history_item(aid):
    r = load_by_id(aid)
    return jsonify(r) if r else (jsonify({"error":"Not found"}), 404)

# ─── Startup ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("\n" + "="*56)
    print("  FairLens v2 — Universal Bias Detection API")
    print("  http://localhost:5000")
    print("  Fairlearn:", "✓ available" if FAIRLEARN_OK else "✗ not installed (using fallback)")
    print("="*56 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
