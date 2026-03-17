@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        f = request.files["file"]
        df_raw = pd.read_csv(f)

        df = clean_dataframe(df_raw)

        # Get optional manual selection from frontend
        sensitive_col = request.form.get("sensitive_col")
        target_col = request.form.get("target_col")

        # Auto-detect if not provided
        sensitive_attrs, auto_target, _ = detect_columns(df)

        if not target_col:
            target_col = auto_target

        if not sensitive_col:
            # pick first detected sensitive attribute
            if sensitive_attrs:
                sensitive_col = list(sensitive_attrs.values())[0]

        if not sensitive_col or not target_col:
            return jsonify({"error": "Could not detect required columns"}), 400

        # Convert target to binary
        df[target_col] = binarize_target(df[target_col])
        df = df.dropna(subset=[target_col])

        # ─── BASELINE MODEL ───
        preds, probas, feature_cols, feature_importance = run_baseline(df, target_col)

        before_metrics = compute_metrics(df, sensitive_col, target_col, preds)

        # ─── MITIGATION ───
        mitigated_preds = run_mitigation(
            df,
            sensitive_col,
            target_col,
            methods=["pre", "post"]
        )

        after_metrics = compute_metrics(df, sensitive_col, target_col, mitigated_preds)

        # ─── IMPROVEMENT ───
        improvement = {
            "fairness_score_change": round(
                after_metrics["fairness_score"] - before_metrics["fairness_score"], 2
            ),
            "disparate_impact_change": round(
                after_metrics["disparate_impact"] - before_metrics["disparate_impact"], 3
            ),
            "accuracy_change": round(
                after_metrics["accuracy"] - before_metrics["accuracy"], 3
            )
        }

        # ─── EXPLANATIONS ───
        explanation = build_explanation(df, sensitive_col, target_col, before_metrics, after_metrics)

        # ─── RECOMMENDATIONS ───
        recommendations = build_recommendations(before_metrics)

        # ─── FEATURE BIAS ───
        feature_bias = compute_feature_bias(df, sensitive_col, target_col, feature_cols)

        # ─── SAVE TO DB ───
        analysis_id = save_to_db(
            f.filename,
            sensitive_col,
            target_col,
            before_metrics,
            after_metrics,
            improvement,
            len(df)
        )

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "sensitive_column": sensitive_col,
            "target_column": target_col,
            "before": before_metrics,
            "after": after_metrics,
            "improvement": improvement,
            "explanation": explanation,
            "recommendations": recommendations,
            "feature_bias": feature_bias,
            "feature_importance": feature_importance
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
