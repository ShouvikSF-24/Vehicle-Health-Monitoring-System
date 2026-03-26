

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, joblib, time
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      confusion_matrix, classification_report,
                                      roc_curve, auc)
from sklearn.preprocessing    import label_binarize

# ─── Optional imports (graceful fallback if not installed) ───────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed. Using GradientBoosting as fallback.")
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers  import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
    tf.get_logger().setLevel("ERROR")
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not installed. Using MLP as fallback for LSTM.")
    from sklearn.neural_network import MLPClassifier

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

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the cleaned CSV files from Step 2."""
    print("📂 Loading cleaned datasets...")
    t_df = pd.read_csv("data/vehicle_telemetry_clean.csv")
    b_df = pd.read_csv("data/battery_health_clean.csv")

    # Use telemetry as primary (it's the larger, richer dataset)
    # We train on it directly rather than merging mismatched columns
    combined = t_df.copy()

    print(f"   Telemetry : {t_df.shape}")
    print(f"   Battery   : {b_df.shape}")
    print(f"   Combined  : {combined.shape}")
    return t_df, b_df, combined


# ════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits data into Training set (80%) and Test set (20%).

    WHY SPLIT?
      We train the model on 80% of data, then TEST how well
      it predicts on the 20% it has NEVER seen before.
      This simulates real-world performance.

    stratify=y ensures the fault ratio is preserved in both splits.
    """
    feature_cols = [c for c in df.columns if c != "fault"]
    X = df[feature_cols].values
    y = df["fault"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y          # keeps same fault % in train and test
    )

    print(f"\n📐 Data Split:")
    print(f"   Train: {X_train.shape[0]} samples "
          f"({(y_train==1).sum()} faults)")
    print(f"   Test : {X_test.shape[0]} samples "
          f"({(y_test==1).sum()} faults)")

    return X_train, X_test, y_train, y_test, feature_cols


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ════════════════════════════════════════════════════════════════════════════
def evaluate_model(name: str, y_true, y_pred, y_prob=None) -> dict:
    """
    Calculates all the metrics we care about:

    Accuracy  = % of ALL predictions that are correct
    Precision = Of all "fault" predictions, how many were ACTUALLY faults?
                (avoids false alarms)
    Recall    = Of all REAL faults, how many did we CATCH?
                (critical for safety — we want this HIGH)
    F1 Score  = Harmonic mean of Precision and Recall. Best single number.
    """
    results = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        results["AUC-ROC"] = round(auc(fpr, tpr), 4)
        results["_fpr"]    = fpr
        results["_tpr"]    = tpr
    else:
        results["AUC-ROC"] = None

    print(f"\n{'─'*50}")
    print(f"  📊 {name} — Evaluation Results")
    print(f"{'─'*50}")
    for k, v in results.items():
        if not k.startswith("_"):
            print(f"   {k:<12}: {v}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal','Fault'])}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — RANDOM FOREST
# ════════════════════════════════════════════════════════════════════════════
def train_random_forest(X_train, y_train, X_test, y_test,
                        feature_names: list) -> dict:
    """
    Random Forest = an ensemble (group) of Decision Trees.

    Decision Tree: A flowchart of yes/no questions.
      e.g., "Is engine_temp > 110°C?" → YES → "Is oil_pressure < 25?" → ...

    Random Forest builds 200 different trees, each trained on a slightly
    different random subset of data. The final prediction is a MAJORITY VOTE.

    ADVANTAGES:
      ✅ Very accurate
      ✅ Handles mixed-scale features well
      ✅ Gives feature importance scores
      ✅ Hard to overfit
    """
    print("\n" + "🌲 " * 15)
    print("  MODEL 1: Random Forest Classifier")
    print("🌲 " * 15)

    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,     # 200 trees in the forest
        max_depth=15,         # Maximum depth of each tree
        min_samples_split=5,  # Need at least 5 samples to split a node
        n_jobs=-1,            # Use all CPU cores
        random_state=42,
        class_weight="balanced"  # Handles imbalanced classes
    )

    rf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"   ⏱️  Training time: {train_time:.2f}s")

    # Cross-validation: train/test on 5 different splits to get reliable score
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
    print(f"   📊 Cross-Val F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Predict
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]  # probability of fault

    # Save model
    joblib.dump(rf, "models/random_forest.pkl")

    # Feature importance plot
    _plot_feature_importance(rf.feature_importances_, feature_names,
                              "Random Forest")

    results = evaluate_model("Random Forest", y_test, y_pred, y_prob)
    results["train_time"] = round(train_time, 2)
    return results, rf, y_pred, y_prob


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — XGBoost (or GradientBoosting fallback)
# ════════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test,
                  feature_names: list) -> dict:
    """
    XGBoost = eXtreme Gradient Boosting.

    BOOSTING means we build trees SEQUENTIALLY:
    1. Train tree 1 → make predictions → note the errors
    2. Train tree 2 to FOCUS on the errors tree 1 made
    3. Train tree 3 to fix what tree 2 missed
    4. Repeat...

    This makes XGBoost very powerful — it keeps learning from mistakes.
    It's the most-used algorithm in data science competitions.

    ADVANTAGES:
      ✅ Often best accuracy
      ✅ Built-in regularization (prevents overfitting)
      ✅ Fast training
    """
    print("\n" + "⚡ " * 15)
    print("  MODEL 2: XGBoost Classifier")
    print("⚡ " * 15)

    start = time.time()
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,     # Small steps → more accurate
            subsample=0.8,          # Use 80% of data per tree
            colsample_bytree=0.8,   # Use 80% of features per tree
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        # Fallback: scikit-learn's GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        print("   (Using GradientBoostingClassifier as XGBoost fallback)")

    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"   ⏱️  Training time: {train_time:.2f}s")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"   📊 Cross-Val F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    joblib.dump(model, "models/xgboost_model.pkl")

    if hasattr(model, "feature_importances_"):
        _plot_feature_importance(model.feature_importances_,
                                 feature_names, "XGBoost")

    model_label = "XGBoost" if HAS_XGB else "GradientBoosting"
    results = evaluate_model(model_label, y_test, y_pred, y_prob)
    results["train_time"] = round(train_time, 2)
    return results, model, y_pred, y_prob


# ════════════════════════════════════════════════════════════════════════════
# MODEL 3 — LSTM (or MLP fallback)
# ════════════════════════════════════════════════════════════════════════════
def train_lstm(X_train, y_train, X_test, y_test) -> dict:
    """
    LSTM = Long Short-Term Memory neural network.

    A regular neural network treats each sensor reading as INDEPENDENT.
    LSTM REMEMBERS previous readings — it understands sequences.

    Think of it like a mechanic who says:
    "The temperature was fine an hour ago, but now it's spiking AND
     the vibration went up too — that PATTERN means something's wrong."

    For LSTM we reshape data into (samples, timesteps, features)
    We treat each row as a single-step time series here.

    ADVANTAGES:
      ✅ Captures temporal patterns
      ✅ Great for streaming sensor data
      ✅ Can detect gradual degradation patterns
    """
    print("\n" + "🧠 " * 15)
    print("  MODEL 3: LSTM Neural Network")
    print("🧠 " * 15)

    start = time.time()

    if HAS_TF:
        # Reshape for LSTM: (samples, timesteps=1, features)
        X_tr = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_te = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        n_features = X_train.shape[1]

        # Build the LSTM architecture
        model = Sequential([
            Input(shape=(1, n_features)),
            LSTM(128, return_sequences=True),   # First LSTM layer
            Dropout(0.3),                        # Randomly drop 30% — prevents overfitting
            LSTM(64, return_sequences=False),    # Second LSTM layer
            Dropout(0.2),
            Dense(32, activation="relu"),        # Fully connected layer
            Dense(1,  activation="sigmoid")      # Output: probability 0→1
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # EarlyStopping stops training when validation loss stops improving
        early_stop = EarlyStopping(monitor="val_loss", patience=5,
                                   restore_best_weights=True)

        # Calculate class weights for imbalanced data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        class_w  = {0: 1.0, 1: neg / pos}

        history = model.fit(
            X_tr, y_train,
            epochs=30, batch_size=64,
            validation_split=0.15,
            class_weight=class_w,
            callbacks=[early_stop],
            verbose=0
        )

        _plot_training_history(history)

        y_prob_raw = model.predict(X_te, verbose=0).flatten()
        y_pred     = (y_prob_raw >= 0.5).astype(int)
        y_prob     = y_prob_raw
        model_label = "LSTM"
        model.save("models/lstm_model.keras" if hasattr(model, "save") else
                   "models/lstm_model_weights")

    else:
        # Fallback: Multi-Layer Perceptron (also a neural network, no LSTM)
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        model_label = "MLP Neural Network"
        print("   (Using MLPClassifier as LSTM fallback)")
        joblib.dump(model, "models/lstm_fallback_mlp.pkl")

    train_time = time.time() - start
    print(f"   ⏱️  Training time: {train_time:.2f}s")

    results = evaluate_model(model_label, y_test, y_pred, y_prob)
    results["train_time"] = round(train_time, 2)
    return results, model, y_pred, y_prob


# ════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _plot_feature_importance(importances, feature_names: list,
                              model_name: str) -> None:
    """Horizontal bar chart of feature importance scores."""
    idx     = np.argsort(importances)
    n_show  = min(15, len(feature_names))
    idx_top = idx[-n_show:]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
    colors  = plt.cm.plasma(np.linspace(0.3, 0.9, n_show))
    ax.barh(range(n_show),
            importances[idx_top],
            color=colors, edgecolor="#333")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in idx_top],
                       color="white")
    ax.set_xlabel("Feature Importance Score", color="white")
    ax.set_title(f"🔍 {model_name} — Feature Importance",
                 color="white", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fname = f"outputs/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"   📈 Saved: {fname}")


def _plot_training_history(history) -> None:
    """Plot loss and accuracy curves during LSTM training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0f1117")

    metrics = [("loss", "Training Loss"), ("accuracy", "Training Accuracy")]
    colors  = [("#ff6b6b", "#ffd93d"), ("#00d4ff", "#6bcb77")]

    for ax, (metric, title), (c_train, c_val) in zip(axes, metrics, colors):
        ax.plot(history.history[metric],
                color=c_train, lw=2, label="Train")
        if f"val_{metric}" in history.history:
            ax.plot(history.history[f"val_{metric}"],
                    color=c_val, lw=2, linestyle="--", label="Validation")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("🧠 LSTM Training History",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/lstm_training_history.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/lstm_training_history.png")


# ════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOTS
# ════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(results_list: list) -> None:
    """
    Confusion Matrix shows:
    ┌─────────────┬──────────────────┬──────────────────┐
    │             │ Predicted Normal │  Predicted Fault │
    ├─────────────┼──────────────────┼──────────────────┤
    │ Actual Norm │  True Negative   │  False Positive  │
    │ Actual Fault│  False Negative  │  True Positive   │
    └─────────────┴──────────────────┴──────────────────┘
    
    False Negatives are DANGEROUS — we missed a real fault!
    We want those to be as LOW as possible.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor="#0f1117")
    if n == 1:
        axes = [axes]

    for ax, (name, y_true, y_pred) in zip(axes, results_list):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=1, linecolor="#1a1d27",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name}\nConfusion Matrix",
                     color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label",  color="white")
        ax.set_ylabel("Actual Label",     color="white")
        ax.set_xticklabels(["Normal", "Fault"], color="white")
        ax.set_yticklabels(["Normal", "Fault"], color="white", rotation=0)

    plt.suptitle("🔲 Confusion Matrices — All Models",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/confusion_matrices.png")


# ════════════════════════════════════════════════════════════════════════════
# ROC CURVES
# ════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(results_with_probs: list) -> None:
    """
    ROC Curve = Receiver Operating Characteristic.
    Plots True Positive Rate vs False Positive Rate at various thresholds.
    
    AUC = Area Under the Curve.
    AUC = 1.0 → Perfect model
    AUC = 0.5 → Random guessing (no better than a coin flip)
    
    We want AUC as close to 1.0 as possible!
    """
    PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1117")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5,
            color="#555", label="Random Guess (AUC = 0.50)")

    for (name, y_true, y_prob), color in zip(results_with_probs, PALETTE):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
    ax.set_title("📈 ROC Curves — All Models",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/roc_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "🤖 " * 20)
    print("  VEHICLE HEALTH MONITORING SYSTEM")
    print("  Step 3: Model Training")
    print("🤖 " * 20)

    # ── Load & prepare data ──────────────────────────────────────────────────
    t_df, b_df, combined_df = load_clean_data()

    # Use combined data for training (more data = better models)
    X_train, X_test, y_train, y_test, feature_cols = split_data(combined_df)

    # ── Train models ─────────────────────────────────────────────────────────
    all_results  = []
    conf_data    = []
    roc_data     = []

    # Model 1: Random Forest
    rf_results, rf_model, rf_pred, rf_prob = train_random_forest(
        X_train, y_train, X_test, y_test, feature_cols)
    all_results.append(rf_results)
    conf_data.append(("Random Forest", y_test, rf_pred))
    roc_data.append(("Random Forest", y_test, rf_prob))

    # Model 2: XGBoost
    xgb_results, xgb_model, xgb_pred, xgb_prob = train_xgboost(
        X_train, y_train, X_test, y_test, feature_cols)
    all_results.append(xgb_results)
    conf_data.append((xgb_results["Model"], y_test, xgb_pred))
    roc_data.append((xgb_results["Model"], y_test, xgb_prob))

    # Model 3: LSTM / MLP
    lstm_results, lstm_model, lstm_pred, lstm_prob = train_lstm(
        X_train, y_train, X_test, y_test)
    all_results.append(lstm_results)
    conf_data.append((lstm_results["Model"], y_test, lstm_pred))
    roc_data.append((lstm_results["Model"], y_test, lstm_prob))

    # ── Visualizations ───────────────────────────────────────────────────────
    print("\n📊 Generating evaluation plots...")
    plot_confusion_matrices(conf_data)
    plot_roc_curves(roc_data)

    # ── Save results for Step 4 ──────────────────────────────────────────────
    results_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if not k.startswith("_")}
                                for r in all_results])
    results_df.to_csv("data/model_results.csv", index=False)
    print("\n📝 Model results saved → data/model_results.csv")

    # Quick preview
    print("\n" + "─" * 60)
    print(results_df.to_string(index=False))
    print("─" * 60)

    # Save test data for Step 4
    import numpy as np
    np.save("data/X_test.npy",  X_test)
    np.save("data/y_test.npy",  y_test)
    np.save("data/rf_prob.npy", rf_prob)
    pd.Series(feature_cols).to_csv("data/feature_cols.csv", index=False)

    print("\n✅ Step 3 Complete! All models trained and saved.")
    print("   ➡️  Next: Run python step4_evaluate_compare.py\n")
