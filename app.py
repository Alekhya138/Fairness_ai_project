"""
FairLens Backend — Full Fairness & Bias Detection Engine
Flask API with Fairlearn, AIF360-style metrics, SQLite storage
Run: pip install flask flask-cors pandas numpy scikit-learn fairlearn && python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import sqlite3
import os
import io
import base64
import traceback
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

# ── Fairlearn (optional but preferred) ────────────────────────────────────────
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
    print("[INFO] Fairlearn available — in-processing mitigation enabled.")
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("[WARN] Fairlearn not installed. Falling back to custom mitigation.")

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), "fairlens.db")

# ── Database ───────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            before_results TEXT,
            after_results TEXT,
            improvement TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(filename, before, after, improvement):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (filename, timestamp, before_results, after_results, improvement)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, datetime.now().isoformat(), json.dumps(before), json.dumps(after), json.dumps(improvement)))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_all_analyses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, timestamp FROM analyses ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "timestamp": r[2]} for r in rows]

def get_analysis_by_id(analysis_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "filename": row[1], "timestamp": row[2],
            "before": json.loads(row[3]),
            "after": json.loads(row[4]),
            "improvement": json.loads(row[5])
        }
    return None


# ── Column Auto-Detection ─────────────────────────────────────────────────────
SENSITIVE_KEYWORDS = {
    "gender":     ["gender", "sex", "male", "female"],
    "race":       ["race", "ethnicity", "ethnic", "race_ethnicity"],
    "age":        ["age", "age_group", "agegroup"],
}
TARGET_KEYWORDS = ["approved", "approval", "outcome", "label", "target", "decision",
                   "hired", "granted", "result", "loan_approved", "credit_approved"]

def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    sensitive = {}
    for attr, keywords in SENSITIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in cols:
                sensitive[attr] = cols[kw]
                break
    target = None
    for kw in TARGET_KEYWORDS:
        if kw in cols:
            target = cols[kw]
            break
    if target is None:
        # Fallback: last binary column
        for col in reversed(df.columns):
            uniq = df[col].dropna().unique()
            if len(uniq) <= 2:
                target = col
                break
    return sensitive, target


# ── Fairness Metrics ──────────────────────────────────────────────────────────
def compute_fairness_metrics(df, sensitive_col, target_col, predictions=None):
    """Compute all fairness metrics for a given sensitive attribute."""
    results = {}
    if predictions is None:
        predictions = df[target_col].values

    groups = df[sensitive_col].unique()
    group_stats = {}
    for g in groups:
        mask = df[sensitive_col] == g
        n = mask.sum()
        approved = predictions[mask.values].sum()
        rate = approved / n if n > 0 else 0
        actual_approved = df.loc[mask, target_col].sum()
        actual_rate = actual_approved / n if n > 0 else 0
        group_stats[str(g)] = {
            "n": int(n),
            "approved": int(approved),
            "approval_rate": round(float(rate), 4),
            "actual_approval_rate": round(float(actual_rate), 4)
        }

    # Overall accuracy
    overall_accuracy = float(np.mean(predictions == df[target_col].values))
    results["accuracy"] = round(overall_accuracy, 4)

    # Statistical Parity Difference (max group rate - min group rate)
    rates = [s["approval_rate"] for s in group_stats.values()]
    max_rate, min_rate = max(rates), min(rates)
    spd = round(max_rate - min_rate, 4)
    results["statistical_parity_difference"] = spd

    # Disparate Impact (min / max rate)
    di = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
    results["disparate_impact"] = di

    # Equal Opportunity (TPR parity) — privileged = highest approval group
    privileged_group = max(group_stats, key=lambda g: group_stats[g]["approval_rate"])
    priv_mask = (df[sensitive_col].astype(str) == privileged_group) & (df[target_col] == 1)
    priv_tp = (predictions[priv_mask.values] == 1).sum()
    priv_pos = priv_mask.sum()
    priv_tpr = priv_tp / priv_pos if priv_pos > 0 else 0

    eo_diffs = []
    for g, s in group_stats.items():
        g_mask = (df[sensitive_col].astype(str) == g) & (df[target_col] == 1)
        g_tp = (predictions[g_mask.values] == 1).sum()
        g_pos = g_mask.sum()
        g_tpr = g_tp / g_pos if g_pos > 0 else 0
        eo_diffs.append(abs(g_tpr - priv_tpr))
    results["equal_opportunity_difference"] = round(float(np.mean(eo_diffs)), 4)

    # Fairness Score (composite, 0-100)
    di_score = min(di / 0.8, 1.0) * 35
    spd_score = max(0, 1 - spd / 0.3) * 35
    eo_score = max(0, 1 - results["equal_opportunity_difference"] / 0.3) * 30
    fairness_score = round(di_score + spd_score + eo_score, 1)
    results["fairness_score"] = fairness_score

    # Bias level
    if fairness_score >= 75:
        results["bias_level"] = "LOW"
        results["bias_color"] = "green"
    elif fairness_score >= 50:
        results["bias_level"] = "MEDIUM"
        results["bias_color"] = "yellow"
    else:
        results["bias_level"] = "HIGH"
        results["bias_color"] = "red"

    results["group_stats"] = group_stats
    results["privileged_group"] = privileged_group
    return results


# ── Bias Explanation Engine (XAI) ────────────────────────────────────────────
def generate_explanation(df, sensitive_col, target_col, metrics_before, metrics_after):
    explanations = []
    group_stats = metrics_before["group_stats"]
    rates = {g: s["approval_rate"] for g, s in group_stats.items()}
    sorted_groups = sorted(rates.items(), key=lambda x: x[1])
    disadvantaged = sorted_groups[0][0]
    privileged = sorted_groups[-1][0]
    gap = round((rates[privileged] - rates[sorted_groups[0][1]]) * 100 if len(sorted_groups) > 1 else 0, 1)

    # Representation check
    total = len(df)
    disadv_count = (df[sensitive_col].astype(str) == disadvantaged).sum()
    disadv_pct = round(disadv_count / total * 100, 1)
    disadv_approval = group_stats[disadvantaged]["actual_approval_rate"] * 100

    if disadv_pct < 35:
        explanations.append({
            "type": "Underrepresentation",
            "icon": "📊",
            "text": f"'{disadvantaged}' group makes up only {disadv_pct}% of the training data. Underrepresentation causes the model to learn biased patterns, systematically disadvantaging this group.",
            "severity": "HIGH"
        })

    # Approval rate gap
    if rates[privileged] > 0:
        rate_gap = (rates[privileged] - rates[disadvantaged]) / rates[privileged] * 100
        if rate_gap > 25:
            explanations.append({
                "type": "Approval Rate Disparity",
                "icon": "⚖️",
                "text": f"'{privileged}' applicants are approved at {rates[privileged]*100:.1f}% vs {rates[disadvantaged]*100:.1f}% for '{disadvantaged}' — a {rate_gap:.1f}% relative gap. This suggests the sensitive attribute '{sensitive_col}' is influencing decisions.",
                "severity": "HIGH"
            })

    # Disparate impact
    di = metrics_before["disparate_impact"]
    if di < 0.8:
        explanations.append({
            "type": "Legal Threshold Violated",
            "icon": "⚠️",
            "text": f"Disparate Impact ratio is {di:.2f}, below the legal 0.80 threshold (EEOC 4/5ths Rule). The model may be in violation of anti-discrimination law in employment, lending, or housing contexts.",
            "severity": "HIGH" if di < 0.6 else "MEDIUM"
        })

    # SPD
    spd = metrics_before["statistical_parity_difference"]
    if spd > 0.15:
        explanations.append({
            "type": "Statistical Parity Failure",
            "icon": "📉",
            "text": f"Statistical Parity Difference is {spd:.2f}. The model applies different standards to different groups. Ideally this should be 0 — meaning equal positive outcome rates regardless of group membership.",
            "severity": "MEDIUM"
        })

    # Improvement explanation
    if metrics_after:
        improvement = metrics_after["fairness_score"] - metrics_before["fairness_score"]
        explanations.append({
            "type": "Mitigation Applied",
            "icon": "✅",
            "text": f"After applying bias mitigation, the fairness score improved by +{improvement:.1f} points. The system now distributes decisions more equitably across all groups.",
            "severity": "IMPROVEMENT"
        })

    return explanations


# ── Ethical Recommendations ───────────────────────────────────────────────────
def generate_recommendations(metrics):
    recs = []
    di = metrics["disparate_impact"]
    spd = metrics["statistical_parity_difference"]
    eo = metrics["equal_opportunity_difference"]
    fs = metrics["fairness_score"]

    if di < 0.8:
        recs.append({
            "priority": "HIGH",
            "icon": "🔴",
            "title": "Collect More Balanced Training Data",
            "detail": f"Disparate Impact is {di:.2f}. Actively collect data from underrepresented groups to ensure the model learns from diverse examples."
        })
    if spd > 0.2:
        recs.append({
            "priority": "HIGH",
            "icon": "🔴",
            "title": "Remove or Encode Sensitive Attributes",
            "detail": f"Statistical Parity Difference ({spd:.2f}) suggests the sensitive attribute has too much influence. Consider removing it or using fairness constraints during training."
        })
    if eo > 0.15:
        recs.append({
            "priority": "MEDIUM",
            "icon": "🟡",
            "title": "Adjust Decision Thresholds Per Group",
            "detail": f"Equal Opportunity Difference is {eo:.2f}. Apply post-processing threshold adjustment to ensure qualified individuals across all groups receive equal treatment."
        })
    if fs < 60:
        recs.append({
            "priority": "MEDIUM",
            "icon": "🟡",
            "title": "Conduct a Full Algorithmic Audit",
            "detail": "Fairness score is critically low. Engage a third-party audit firm to review model features, training data, and deployment context before production use."
        })
    recs.append({
        "priority": "LOW",
        "icon": "🟢",
        "title": "Implement Ongoing Monitoring",
        "detail": "Deploy fairness monitoring in production. Data distributions shift over time and bias can re-emerge. Set automated alerts when fairness metrics degrade."
    })
    recs.append({
        "priority": "LOW",
        "icon": "🟢",
        "title": "Document Model Cards & Data Sheets",
        "detail": "Create a model card documenting known biases, intended use cases, and fairness evaluations following the EU AI Act and NIST AI RMF guidelines."
    })
    return recs


# ── Pre-processing Mitigation ─────────────────────────────────────────────────
def apply_reweighing(df, sensitive_col, target_col):
    """Reweigh samples to equalize representation across groups and outcomes."""
    df = df.copy()
    n = len(df)
    group_outcome_counts = df.groupby([sensitive_col, target_col]).size()
    weights = np.ones(n)
    groups = df[sensitive_col].unique()
    outcomes = df[target_col].unique()
    for g in groups:
        for o in outcomes:
            expected = (df[sensitive_col] == g).sum() / n * (df[target_col] == o).sum() / n
            observed = group_outcome_counts.get((g, o), 1) / n
            w = expected / observed if observed > 0 else 1.0
            mask = (df[sensitive_col] == g) & (df[target_col] == o)
            weights[mask.values] = w
    df["_weight"] = weights
    return df


# ── In-processing Mitigation ──────────────────────────────────────────────────
def train_fair_model(X_train, y_train, sensitive_train, X_test, sensitive_col_name):
    """Train fairness-aware model using Fairlearn or adversarial approach."""
    if FAIRLEARN_AVAILABLE:
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(base_clf, constraint)
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        preds = mitigator.predict(X_test)
        return preds
    else:
        # Custom adversarial-style: upsample minority groups
        return None  # Falls back to post-processing only


# ── Post-processing Mitigation ────────────────────────────────────────────────
def apply_threshold_adjustment(df, sensitive_col, target_col, probas, threshold_default=0.5):
    """Adjust per-group decision thresholds to equalize approval rates."""
    groups = df[sensitive_col].unique()
    overall_rate = df[target_col].mean()
    predictions = np.zeros(len(df), dtype=int)
    for g in groups:
        mask = df[sensitive_col] == g
        group_probas = probas[mask.values]
        # Find threshold that brings this group's rate close to overall rate
        best_thresh = threshold_default
        best_diff = float("inf")
        for thresh in np.arange(0.3, 0.9, 0.02):
            preds_g = (group_probas >= thresh).astype(int)
            rate = preds_g.mean()
            diff = abs(rate - overall_rate)
            if diff < best_diff:
                best_diff = diff
                best_thresh = thresh
        predictions[mask.values] = (group_probas >= best_thresh).astype(int)
    return predictions


# ── Full Mitigation Pipeline ──────────────────────────────────────────────────
def run_mitigation_pipeline(df, sensitive_col, target_col, methods=None):
    """Run full 3-stage mitigation and return predictions + metrics."""
    if methods is None:
        methods = ["pre", "in", "post"]

    df_work = df.copy()
    le = LabelEncoder()

    # Encode categorical columns
    encoders = {}
    for col in df_work.select_dtypes(include=["object"]).columns:
        if col != target_col:
            enc = LabelEncoder()
            df_work[col + "_enc"] = enc.fit_transform(df_work[col].astype(str))
            encoders[col] = enc

    # Features
    feature_cols = [c for c in df_work.columns if c not in [target_col, sensitive_col, "_weight"]
                    and not df_work[c].dtype == object]

    X = df_work[feature_cols].fillna(0).values
    y = df_work[target_col].values
    sensitive = df_work[sensitive_col].astype(str).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stage 1: Pre-processing (Reweighing)
    sample_weights = None
    if "pre" in methods:
        df_reweighed = apply_reweighing(df_work, sensitive_col, target_col)
        sample_weights = df_reweighed["_weight"].values

    # Train base model for probabilities
    clf = LogisticRegression(max_iter=1000, random_state=42)
    if sample_weights is not None:
        clf.fit(X_scaled, y, sample_weight=sample_weights)
    else:
        clf.fit(X_scaled, y)

    probas = clf.predict_proba(X_scaled)[:, 1]

    # Stage 2: In-processing (Fairlearn or custom)
    if "in" in methods and FAIRLEARN_AVAILABLE:
        try:
            preds_fair = train_fair_model(X_scaled, y, sensitive, X_scaled, sensitive_col)
            if preds_fair is not None:
                return preds_fair
        except Exception as e:
            print(f"[WARN] Fairlearn in-processing failed: {e}")

    # Stage 3: Post-processing (Threshold adjustment)
    if "post" in methods:
        predictions = apply_threshold_adjustment(df_work, sensitive_col, target_col, probas)
    else:
        predictions = (probas >= 0.5).astype(int)

    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "fairlearn": FAIRLEARN_AVAILABLE, "timestamp": datetime.now().isoformat()})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint — detects bias, optionally mitigates."""
    try:
        # Parse CSV
        if "file" in request.files:
            f = request.files["file"]
            filename = f.filename
            df = pd.read_csv(f)
        elif "dataset" in request.form:
            dataset_name = request.form["dataset"]
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{dataset_name}.csv")
            df = pd.read_csv(csv_path)
            filename = f"{dataset_name}.csv"
        else:
            return jsonify({"error": "No file or dataset provided"}), 400

        # Auto-detect columns
        sensitive_map, target_col = detect_columns(df)
        if not target_col:
            return jsonify({"error": "Could not detect target column. Please ensure your CSV has an 'approved' or 'outcome' column."}), 400

        # Use user-specified attribute or first detected
        attr_override = request.form.get("sensitive_attr", "")
        if attr_override and attr_override in df.columns:
            sensitive_col = attr_override
        elif sensitive_map:
            sensitive_col = list(sensitive_map.values())[0]
        else:
            return jsonify({"error": "Could not detect sensitive attribute column."}), 400

        # Ensure target is binary integer
        df[target_col] = df[target_col].apply(lambda x: 1 if str(x).lower() in ["1", "true", "yes", "approved", "hired"] else 0)

        # ── BEFORE: Baseline predictions (simple LR) ──────────────────────────
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object"]).columns:
            if col != target_col:
                df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        feature_cols = [c for c in df_enc.columns if c not in [target_col]]
        X = df_enc[feature_cols].fillna(0).values
        y = df_enc[target_col].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        base_clf.fit(X_s, y)
        before_preds = base_clf.predict(X_s)

        metrics_before = compute_fairness_metrics(df, sensitive_col, target_col, before_preds)

        # ── AFTER: Mitigated predictions ──────────────────────────────────────
        mitigation_methods = request.form.get("mitigation", "pre,in,post").split(",")
        after_preds = run_mitigation_pipeline(df, sensitive_col, target_col, mitigation_methods)
        metrics_after = compute_fairness_metrics(df, sensitive_col, target_col, after_preds)

        # ── Improvement calculation ───────────────────────────────────────────
        improvement = {
            "fairness_score": round(metrics_after["fairness_score"] - metrics_before["fairness_score"], 1),
            "disparate_impact": round(metrics_after["disparate_impact"] - metrics_before["disparate_impact"], 4),
            "statistical_parity": round(metrics_before["statistical_parity_difference"] - metrics_after["statistical_parity_difference"], 4),
            "equal_opportunity": round(metrics_before["equal_opportunity_difference"] - metrics_after["equal_opportunity_difference"], 4),
            "pct_improvement": round((metrics_after["fairness_score"] - metrics_before["fairness_score"]) / max(100 - metrics_before["fairness_score"], 1) * 100, 1)
        }

        # ── XAI Explanations ──────────────────────────────────────────────────
        explanations = generate_explanation(df, sensitive_col, target_col, metrics_before, metrics_after)

        # ── Ethical Recommendations ───────────────────────────────────────────
        recommendations = generate_recommendations(metrics_before)

        # ── Column Info ───────────────────────────────────────────────────────
        col_info = {
            "sensitive_col": sensitive_col,
            "target_col": target_col,
            "detected_attributes": sensitive_map,
            "total_rows": len(df),
            "columns": list(df.columns)
        }

        # ── What-If Data ──────────────────────────────────────────────────────
        # For simulator: store feature importances
        try:
            feat_imp = dict(zip(feature_cols, np.abs(base_clf.coef_[0])))
            top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
        except:
            top_features = []

        # ── Save to DB ────────────────────────────────────────────────────────
        analysis_id = save_analysis(filename, metrics_before, metrics_after, improvement)

        return jsonify({
            "analysis_id": analysis_id,
            "column_info": col_info,
            "before": metrics_before,
            "after": metrics_after,
            "improvement": improvement,
            "explanations": explanations,
            "recommendations": recommendations,
            "top_features": top_features,
            "mitigation_applied": mitigation_methods,
            "dataset_preview": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/whatif", methods=["POST"])
def whatif():
    """What-If simulator: predict outcome for custom input, test fairness."""
    try:
        data = request.get_json()
        inputs = data.get("inputs", {})
        sensitive_attr = data.get("sensitive_attr", "gender")

        # Simulate with stored model coefficients (simplified)
        # In production: load the trained model from session/DB
        age = float(inputs.get("age", 35))
        income = float(inputs.get("income", 55000))
        credit_score = float(inputs.get("credit_score", 680))
        employment_years = float(inputs.get("employment_years", 5))
        gender = inputs.get("gender", "Male")

        def simple_predict(age, income, credit, emp, gender_val):
            score = (income / 100000 * 0.4 +
                     credit / 850 * 0.3 +
                     emp / 20 * 0.15 +
                     age / 60 * 0.1 +
                     (0.05 if gender_val == "Male" else -0.05))  # Intentional bias for demo
            prob = 1 / (1 + np.exp(-10 * (score - 0.5)))
            return round(float(prob), 4)

        # Original prediction
        prob_original = simple_predict(age, income, credit_score, employment_years, gender)
        pred_original = 1 if prob_original >= 0.5 else 0

        # Counterfactual: flip gender
        counterfactual_gender = "Female" if gender == "Male" else "Male"
        prob_counterfactual = simple_predict(age, income, credit_score, employment_years, counterfactual_gender)
        pred_counterfactual = 1 if prob_counterfactual >= 0.5 else 0

        gender_matters = pred_original != pred_counterfactual

        return jsonify({
            "prediction": pred_original,
            "probability": prob_original,
            "outcome": "Approved ✓" if pred_original == 1 else "Denied ✗",
            "counterfactual": {
                "gender_changed_to": counterfactual_gender,
                "prediction": pred_counterfactual,
                "probability": prob_counterfactual,
                "outcome": "Approved ✓" if pred_counterfactual == 1 else "Denied ✗",
                "gender_affected_result": gender_matters,
                "alert": f"⚠️ Changing ONLY gender to '{counterfactual_gender}' changes the decision from '{'Approved' if pred_original else 'Denied'}' to '{'Approved' if pred_counterfactual else 'Denied'}'. This indicates gender bias." if gender_matters else f"✓ Changing gender does not affect this applicant's outcome."
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    """Return all past analyses."""
    return jsonify(get_all_analyses())


@app.route("/api/history/<int:analysis_id>", methods=["GET"])
def get_history(analysis_id):
    """Return a specific past analysis."""
    result = get_analysis_by_id(analysis_id)
    if result:
        return jsonify(result)
    return jsonify({"error": "Analysis not found"}), 404


@app.route("/api/demo", methods=["GET"])
def demo():
    """Run analysis on the preloaded demo dataset."""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "loan_demo.csv")
        df = pd.read_csv(csv_path)
        sensitive_col = "gender"
        target_col = "approved"

        # Encode
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=["object"]).columns:
            if col != target_col:
                df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        feature_cols = [c for c in df_enc.columns if c != target_col]
        X = df_enc[feature_cols].fillna(0).values
        y = df_enc[target_col].values
        X_s = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_s, y)
        before_preds = clf.predict(X_s)
        metrics_before = compute_fairness_metrics(df, sensitive_col, target_col, before_preds)
        after_preds = run_mitigation_pipeline(df, sensitive_col, target_col, ["pre", "post"])
        metrics_after = compute_fairness_metrics(df, sensitive_col, target_col, after_preds)
        improvement = {
            "fairness_score": round(metrics_after["fairness_score"] - metrics_before["fairness_score"], 1),
            "disparate_impact": round(metrics_after["disparate_impact"] - metrics_before["disparate_impact"], 4),
            "statistical_parity": round(metrics_before["statistical_parity_difference"] - metrics_after["statistical_parity_difference"], 4),
            "equal_opportunity": round(metrics_before["equal_opportunity_difference"] - metrics_after["equal_opportunity_difference"], 4),
            "pct_improvement": round((metrics_after["fairness_score"] - metrics_before["fairness_score"]) / max(100 - metrics_before["fairness_score"], 1) * 100, 1)
        }
        explanations = generate_explanation(df, sensitive_col, target_col, metrics_before, metrics_after)
        recommendations = generate_recommendations(metrics_before)

        return jsonify({
            "analysis_id": 0,
            "column_info": {"sensitive_col": "gender", "target_col": "approved", "total_rows": len(df), "columns": list(df.columns)},
            "before": metrics_before,
            "after": metrics_after,
            "improvement": improvement,
            "explanations": explanations,
            "recommendations": recommendations,
            "mitigation_applied": ["pre", "post"]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Start Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("\n" + "="*60)
    print("  FairLens Backend Server")
    print("  http://localhost:5000")
    print("  API: http://localhost:5000/api/health")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)