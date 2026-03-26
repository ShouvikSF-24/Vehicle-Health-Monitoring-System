

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

# ─── Plotting style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "grid.color":       "#2a2d3a",
})
PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]

os.makedirs("outputs", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD SAVED RESULTS
# ════════════════════════════════════════════════════════════════════════════
def load_results():
    """Loads model results and test data saved in Step 3."""
    print("📂 Loading results from Step 3...")
    results_df   = pd.read_csv("data/model_results.csv")
    X_test       = np.load("data/X_test.npy")
    y_test       = np.load("data/y_test.npy")
    feature_cols = pd.read_csv("data/feature_cols.csv").iloc[:, 0].tolist()
    rf_model     = joblib.load("models/random_forest.pkl")
    xgb_model    = joblib.load("models/xgboost_model.pkl")
    print(f"   Results loaded: {results_df.shape}")
    return results_df, X_test, y_test, feature_cols, rf_model, xgb_model


# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE (pretty print + plot)
# ════════════════════════════════════════════════════════════════════════════
def create_comparison_table(results_df: pd.DataFrame) -> None:
    """
    Creates a visual comparison table showing all metrics side by side.
    Also highlights the WINNER for each metric.
    """
    print("\n" + "=" * 65)
    print("  🏆 MODEL COMPARISON TABLE")
    print("=" * 65)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    display_df = results_df[["Model"] + metrics].copy()
    print(display_df.to_string(index=False))

    # Identify the best model per metric
    print("\n  🥇 Best per metric:")
    for metric in metrics:
        col = display_df[metric].dropna()
        if not col.empty:
            best_idx   = col.idxmax()
            best_model = display_df.loc[best_idx, "Model"]
            best_val   = col.max()
            print(f"   {metric:<12}: {best_model:<22} ({best_val:.4f})")

    # Overall winner (by F1 Score)
    best_overall = display_df.loc[display_df["F1 Score"].idxmax(), "Model"]
    print(f"\n  🏆 Overall Best Model (F1): {best_overall}")

    # ── Save as styled PNG ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f1117")
    ax.axis("off")

    col_labels = ["Model"] + metrics + ["Train Time (s)"]
    if "train_time" in results_df.columns:
        cell_data = [[row["Model"]] +
                     [f"{row[m]:.4f}" if pd.notnull(row.get(m)) else "N/A"
                      for m in metrics] +
                     [f"{row.get('train_time', 'N/A')}"]
                     for _, row in results_df.iterrows()]
    else:
        cell_data = [[row["Model"]] +
                     [f"{row[m]:.4f}" if pd.notnull(row.get(m)) else "N/A"
                      for m in metrics]
                     for _, row in results_df.iterrows()]
        col_labels = ["Model"] + metrics

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1e3a5f")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style data rows
    row_colors = ["#1a1d27", "#222533"]
    for i in range(1, len(cell_data) + 1):
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(row_colors[(i - 1) % 2])
            table[(i, j)].set_text_props(color="#e0e0e0")

    ax.set_title("🏆 Model Performance Comparison",
                 color="white", fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison_table.png", dpi=130,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("\n   📈 Saved: outputs/model_comparison_table.png")


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════════════════
def plot_comparison_bars(results_df: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing all metrics across all three models.
    Easier to visually see which model wins on which metric.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    models  = results_df["Model"].tolist()
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0f1117")

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        vals = [results_df[results_df["Model"] == model][m].values[0]
                if not pd.isna(results_df[results_df["Model"] == model][m].values[0])
                else 0
                for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=model, color=color,
                        alpha=0.9, edgecolor="#333")

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=11)
    ax.set_ylabel("Score (0–1)", color="white", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("📊 Model Comparison — All Metrics",
                 color="white", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.9, color="#ffd93d", linestyle="--",
               alpha=0.5, label="0.90 target")

    plt.tight_layout()
    plt.savefig("outputs/model_comparison_bars.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/model_comparison_bars.png")


# ════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
def compute_shap_analysis(rf_model, X_test, feature_cols: list,
                           n_samples: int = 500) -> None:
    """
    SHAP = SHapley Additive exPlanations
    (Named after Lloyd Shapley, a Nobel Prize-winning game theorist)

    WHAT IT DOES:
      For EACH prediction, SHAP tells you HOW MUCH each feature
      pushed the probability up or down.

    EXAMPLE:
      Vehicle X predicted as FAULT (prob = 0.87)
      SHAP says:
        engine_temp    → +0.42  (pushed fault probability UP a lot)
        oil_pressure   → +0.28  (also pushed it up)
        fuel_level     → -0.05  (pushed it down slightly)
        rpm            → +0.12  (pushed up)

    This makes the model EXPLAINABLE — crucial for safety-critical systems!
    """
    try:
        import shap
        print("\n🔍 Computing SHAP values (this may take ~30 seconds)...")

        # Use a subset for speed
        X_sample = X_test[:n_samples]

        # TreeExplainer is fast for tree-based models
        explainer   = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
        if sv.ndim != 2:
            sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        # ── Plot 1: SHAP Summary (beeswarm) ──────────────────────────────────
        # Shows the distribution of SHAP values for each feature
        # Red dots = high feature value, Blue = low
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
        shap.summary_plot(sv, X_sample,
                          feature_names=feature_cols,
                          show=False, plot_type="dot",
                          color_bar=True)
        plt.title("🔍 SHAP Summary Plot — Random Forest",
                  color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("outputs/shap_summary.png", dpi=120,
                    bbox_inches="tight", facecolor="#0f1117")
        plt.close()
        print("   📈 Saved: outputs/shap_summary.png")

        # ── Plot 2: SHAP Feature Importance (mean absolute) ───────────────────
        mean_shap = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(mean_shap)))
        ax.barh(range(len(mean_shap)),
                mean_shap[sorted_idx],
                color=colors, edgecolor="#333")
        ax.set_yticks(range(len(mean_shap)))
        ax.set_yticklabels([feature_cols[i] for i in sorted_idx],
                           color="white")
        ax.set_xlabel("Mean |SHAP value| (average impact on fault probability)",
                      color="white")
        ax.set_title("🔍 SHAP Feature Importance — Random Forest",
                     color="white", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/shap_importance.png", dpi=120,
                    bbox_inches="tight", facecolor="#0f1117")
        plt.close()
        print("   📈 Saved: outputs/shap_importance.png")

        # ── Waterfall for one prediction ──────────────────────────────────────
        print("\n📋 SHAP Explanation for a Single Prediction:")
        sample_idx = 0   # First test sample
        fault_shap = sv[sample_idx]
        base_val   = explainer.expected_value[1] if isinstance(
            explainer.expected_value, list) else explainer.expected_value

        contrib_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value":   X_sample[sample_idx],
            "SHAP":    fault_shap
        }).sort_values("SHAP", key=abs, ascending=False).head(10)

        print(contrib_df.to_string(index=False))

    except ImportError:
        print("   ⚠️  SHAP not installed.")
        print("   📦 Install with: pip install shap")
        print("   Running manual feature importance instead...\n")
        _manual_importance_fallback(rf_model, feature_cols)


def _manual_importance_fallback(rf_model, feature_cols: list) -> None:
    """Fallback when SHAP isn't available: use RF built-in importance."""
    importances = rf_model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    print("📊 Top 10 Features by Random Forest Importance:")
    for i in sorted_idx[:10]:
        bar = "█" * int(importances[i] * 200)
        print(f"   {feature_cols[i]:<25} {importances[i]:.4f}  {bar}")

    # Also make a simple plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
    top_n = min(12, len(feature_cols))
    top_idx = sorted_idx[:top_n][::-1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n),
            importances[top_idx],
            color=colors, edgecolor="#333")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in top_idx], color="white")
    ax.set_xlabel("Feature Importance Score", color="white")
    ax.set_title("🔍 Feature Importance (RF) — SHAP Fallback",
                 color="white", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_rf_fallback.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/feature_importance_rf_fallback.png")


# ════════════════════════════════════════════════════════════════════════════
# FAULT PROBABILITY PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def predict_vehicle_fault(model, feature_names: list,
                           sensor_readings: dict) -> None:
    """
    THE FINAL FUNCTION — use this in production!
    
    Give it sensor readings from a vehicle, and it returns:
      - Whether a fault is predicted
      - The probability (0–100%) of a fault
      - Which sensors are most concerning

    Parameters:
      model         : The trained ML model (Random Forest, XGBoost, etc.)
      feature_names : List of feature names the model was trained on
      sensor_readings: Dictionary of {feature_name: value}

    Example Usage:
      predict_vehicle_fault(
          model=rf_model,
          feature_names=feature_cols,
          sensor_readings={
              "engine_temp": 135.0,
              "oil_pressure": 22.5,
              ...
          }
      )
    """
    print("\n" + "🔮 " * 20)
    print("  FAULT PROBABILITY PREDICTION")
    print("🔮 " * 20)

    # Build feature vector in the correct order
    feature_vector = np.array([
        sensor_readings.get(f, 0.0) for f in feature_names
    ]).reshape(1, -1)

    # Predict
    pred  = model.predict(feature_vector)[0]
    proba = model.predict_proba(feature_vector)[0]
    fault_prob  = proba[1] * 100
    normal_prob = proba[0] * 100

    # Display
    print("\n📡 Input Sensor Readings:")
    for k, v in sensor_readings.items():
        if k in feature_names:
            print(f"   {k:<25}: {v:.3f}")

    print(f"\n📊 Prediction Results:")
    print(f"   Prediction      : {'🚨 FAULT DETECTED' if pred == 1 else '✅ NORMAL'}")
    print(f"   Fault Probability: {fault_prob:.1f}%")
    print(f"   Normal Probability: {normal_prob:.1f}%")

    # Risk level
    if fault_prob >= 75:
        risk = "🔴 HIGH RISK   — Immediate maintenance required!"
    elif fault_prob >= 50:
        risk = "🟠 MEDIUM RISK — Schedule maintenance soon."
    elif fault_prob >= 25:
        risk = "🟡 LOW RISK    — Monitor closely."
    else:
        risk = "🟢 VERY LOW    — Vehicle operating normally."

    print(f"   Risk Level      : {risk}")

    # ASCII probability bar
    filled = int(fault_prob / 2)
    bar = "█" * filled + "░" * (50 - filled)
    print(f"\n   Fault Probability: [{bar}] {fault_prob:.1f}%")

    return {"prediction": int(pred), "fault_probability": fault_prob,
            "normal_probability": normal_prob}


# ════════════════════════════════════════════════════════════════════════════
# PLOT FAULT PROBABILITY DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
def plot_probability_distribution(rf_model, X_test, y_test,
                                   feature_cols: list) -> None:
    """
    Shows how the model distributes fault probabilities.
    A good model will clearly separate normal (low prob) from fault (high prob).
    """
    proba = rf_model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0f1117")
    bins = np.linspace(0, 1, 51)

    ax.hist(proba[y_test == 0], bins=bins, alpha=0.75,
            color=PALETTE[0], label="Actual: Normal")
    ax.hist(proba[y_test == 1], bins=bins, alpha=0.75,
            color=PALETTE[1], label="Actual: Fault")
    ax.axvline(x=0.5, color=PALETTE[2], linestyle="--",
               lw=2, label="Decision Threshold (0.5)")
    ax.set_xlabel("Predicted Fault Probability", color="white", fontsize=12)
    ax.set_ylabel("Number of Samples",           color="white", fontsize=12)
    ax.set_title("🎯 Fault Probability Distribution — Random Forest",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text zones
    ax.text(0.15, ax.get_ylim()[1] * 0.9, "NORMAL ZONE",
            color=PALETTE[0], fontsize=12, fontweight="bold", ha="center")
    ax.text(0.80, ax.get_ylim()[1] * 0.9, "FAULT ZONE",
            color=PALETTE[1], fontsize=12, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig("outputs/probability_distribution.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/probability_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "📊 " * 20)
    print("  VEHICLE HEALTH MONITORING SYSTEM")
    print("  Step 4: Evaluation & Explainability")
    print("📊 " * 20)

    # Load everything
    results_df, X_test, y_test, feature_cols, rf_model, xgb_model = \
        load_results()

    # ── Comparison ───────────────────────────────────────────────────────────
    print("\n📊 Creating comparison visualizations...")
    create_comparison_table(results_df)
    plot_comparison_bars(results_df)
    plot_probability_distribution(rf_model, X_test, y_test, feature_cols)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    compute_shap_analysis(rf_model, X_test, feature_cols)

    # ── Live prediction demo ─────────────────────────────────────────────────
    # Demo 1: Faulty vehicle (high engine temp, low oil pressure)
    predict_vehicle_fault(
        model=rf_model,
        feature_names=feature_cols,
        sensor_readings={
            "engine_temp":       135.0,   # ⚠️  Way too hot (normal ~90°C)
            "rpm":               3200.0,
            "vehicle_speed":     65.0,
            "oil_pressure":      22.5,    # ⚠️  Too low (normal ~40 PSI)
            "fuel_level":        45.0,
            "coolant_temp":      115.0,   # ⚠️  Overheating
            "throttle_position": 55.0,
            "brake_pressure":    28.0,
            "vibration":         0.95,    # ⚠️  High vibration
            "exhaust_temp":      480.0,   # ⚠️  Too hot
        }
    )

    # Demo 2: Healthy vehicle (all normal readings)
    predict_vehicle_fault(
        model=rf_model,
        feature_names=feature_cols,
        sensor_readings={
            "engine_temp":       88.0,
            "rpm":               2400.0,
            "vehicle_speed":     70.0,
            "oil_pressure":      42.0,
            "fuel_level":        75.0,
            "coolant_temp":      82.0,
            "throttle_position": 40.0,
            "brake_pressure":    30.0,
            "vibration":         0.28,
            "exhaust_temp":      340.0,
        }
    )

    print("\n" + "=" * 65)
    print("  ✅ FULL PIPELINE COMPLETE!")
    print("=" * 65)
    print("\n  📁 All outputs saved to /outputs folder:")
    for f in sorted(os.listdir("outputs")):
        print(f"     📈 {f}")
    print("\n  📁 Models saved to /models folder:")
    for f in sorted(os.listdir("models")):
        print(f"     🤖 {f}")
    print()
