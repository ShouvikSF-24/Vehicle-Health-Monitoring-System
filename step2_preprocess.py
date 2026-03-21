"""
============================================================
 VEHICLE HEALTH MONITORING SYSTEM
 Step 2: Data Preprocessing
============================================================

WHAT THIS FILE DOES:
  - Loads the raw CSV datasets generated in Step 1
  - Cleans the data:
      ✅ Handles missing values
      ✅ Removes outliers
      ✅ Normalizes/scales features
      ✅ Selects the most important features
  - Saves cleaned, ready-to-train data
  - Creates visualizations so you can SEE the data

WHY PREPROCESSING?
  Raw data is messy. ML models are like picky chefs —
  they only cook well with clean, well-prepared ingredients.
  Garbage in = garbage out!

BEGINNER TIP:
  Run after step1. Command: python step2_preprocess.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # Non-interactive backend (saves files)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import os, warnings
warnings.filterwarnings("ignore")

# ─── Styling ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "grid.color":       "#2a2d3a",
    "grid.alpha":       0.5,
})
PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]

os.makedirs("outputs", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the two CSV files we created in Step 1."""
    print("📂 Loading datasets...")
    telemetry_df = pd.read_csv("data/vehicle_telemetry.csv")
    battery_df   = pd.read_csv("data/battery_health.csv")
    print(f"   Telemetry : {telemetry_df.shape}")
    print(f"   Battery   : {battery_df.shape}")
    return telemetry_df, battery_df


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1: Distribution of ALL features (before cleaning)
# This shows you what each sensor's values look like as a histogram
# ════════════════════════════════════════════════════════════════════════════
def plot_distributions(df: pd.DataFrame, title: str, filename: str) -> None:
    """
    Plots a histogram for every feature column.
    Histogram = bar chart of how often each value range appears.
    """
    feature_cols = [c for c in df.columns if c != "fault"]
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(16, n_rows * 3.5),
                             facecolor="#0f1117")
    fig.suptitle(f"📊 {title} — Feature Distributions",
                 fontsize=16, color="white", fontweight="bold", y=1.01)

    axes_flat = axes.flatten()
    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        normal = df[df["fault"] == 0][col].dropna()
        faulty = df[df["fault"] == 1][col].dropna()
        ax.hist(normal, bins=40, alpha=0.7, color=PALETTE[0], label="Normal")
        ax.hist(faulty, bins=40, alpha=0.7, color=PALETTE[1], label="Fault")
        ax.set_title(col, fontsize=10, color="white")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot panels
    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=120, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"   📈 Saved: outputs/{filename}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2: Correlation Heatmap
# Shows which sensors are related to each other and to the fault label
# ════════════════════════════════════════════════════════════════════════════
def plot_correlation(df: pd.DataFrame, title: str, filename: str) -> None:
    """
    Correlation = how much two things move together.
    +1 = perfectly related, -1 = opposite, 0 = no relation.
    """
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # Only show lower half

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0f1117")
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, linecolor="#1a1d27",
                annot_kws={"size": 8})
    ax.set_title(f"🔥 {title} — Correlation Heatmap",
                 fontsize=14, color="white", fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha="right", color="white")
    ax.set_yticklabels(ax.get_yticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=120, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"   📈 Saved: outputs/{filename}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3: Class balance (how many normal vs faulty)
# ════════════════════════════════════════════════════════════════════════════
def plot_class_balance(t_df: pd.DataFrame, b_df: pd.DataFrame) -> None:
    """Shows whether the dataset is balanced (equal normal/fault samples)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f1117")

    for ax, df, name in zip(axes,
                             [t_df, b_df],
                             ["Telemetry", "Battery"]):
        counts = df["fault"].value_counts().sort_index()
        bars = ax.bar(["Normal (0)", "Fault (1)"],
                      counts.values,
                      color=[PALETTE[0], PALETTE[1]],
                      edgecolor="#333", linewidth=1.2, width=0.5)

        # Add count labels on top of bars
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30,
                    f"{count}\n({count/len(df)*100:.1f}%)",
                    ha="center", va="bottom", color="white", fontsize=11)

        ax.set_title(f"{name} — Class Balance",
                     color="white", fontsize=13, fontweight="bold")
        ax.set_ylabel("Sample Count", color="white")
        ax.set_ylim(0, max(counts.values) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("🎯 Dataset Class Distribution",
                 fontsize=15, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/class_balance.png", dpi=120,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("   📈 Saved: outputs/class_balance.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP A: Handle Missing Values
# ════════════════════════════════════════════════════════════════════════════
def handle_missing_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Missing values (NaN) crash most ML models.
    Strategy: Fill each missing value with the MEDIAN of that column.
    Median = the middle value when sorted (more robust than mean for outliers).
    """
    missing_before = df.isnull().sum().sum()

    feature_cols = [c for c in df.columns if c != "fault"]
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    missing_after = df.isnull().sum().sum()
    print(f"   [{name}] Missing values: {missing_before} → {missing_after} ✅")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP B: Remove Outliers using IQR method
# ════════════════════════════════════════════════════════════════════════════
def remove_outliers(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Outliers are extreme values that can mislead the model.
    IQR method: Remove rows where a value is WAY above or below the typical range.
    
    IQR = Interquartile Range = middle 50% of data
    We keep values within: [Q1 - 3*IQR, Q3 + 3*IQR]
    """
    rows_before = len(df)
    feature_cols = [c for c in df.columns if c != "fault"]

    mask = pd.Series([True] * len(df))
    for col in feature_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        mask = mask & (df[col] >= lower) & (df[col] <= upper)

    df_clean = df[mask].reset_index(drop=True)
    removed  = rows_before - len(df_clean)
    print(f"   [{name}] Outliers removed: {removed} rows "
          f"({removed/rows_before*100:.1f}%)  →  {len(df_clean)} rows remain")
    return df_clean


# ════════════════════════════════════════════════════════════════════════════
# STEP C: Normalize / Scale Features
# ════════════════════════════════════════════════════════════════════════════
def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Different sensors have wildly different scales:
      - RPM: 0–5000
      - Temperature: 0–150°C
      - Oil pressure: 0–60 PSI

    ML models get confused by huge differences in scale.
    StandardScaler transforms each feature so:
      - Mean  = 0
      - Std   = 1
    This puts everything on the same playing field.
    """
    feature_cols = [c for c in df.columns if c != "fault"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


# ════════════════════════════════════════════════════════════════════════════
# STEP D: Feature Engineering
# Creating NEW useful features from existing ones
# ════════════════════════════════════════════════════════════════════════════
def engineer_features_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering = creating smart new inputs that help the model learn.
    Think of it as giving the model extra clues.
    
    We create these new features BEFORE scaling, so we re-normalize after.
    """
    # Thermal stress = combined heat effect
    df["thermal_stress"] = df["engine_temp"] + df["coolant_temp"] * 0.5

    # RPM to speed ratio — engine working too hard?
    df["rpm_speed_ratio"] = df["rpm"] / (df["vehicle_speed"].abs() + 1)

    # Rolling z-score for vibration (how unusual is this vibration?)
    df["vibration_zscore"] = (
        (df["vibration"] - df["vibration"].mean()) / df["vibration"].std()
    )

    print("   [Telemetry] Engineered features: "
          "thermal_stress, rpm_speed_ratio, vibration_zscore")
    return df


def engineer_features_battery(df: pd.DataFrame) -> pd.DataFrame:
    """New features for the battery dataset."""
    # Power = Voltage × Current (Watts)
    df["power"]         = df["voltage"] * df["current"]

    # Degradation index — old + high resistance = bad battery
    df["degradation"]   = df["age_months"] * df["internal_resist"]

    # SOC efficiency — is it delivering power proportional to charge?
    df["soc_efficiency"] = df["power_output"] / (df["state_of_charge"] + 1)

    print("   [Battery] Engineered features: power, degradation, soc_efficiency")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP E: Feature Selection using SelectKBest
# ════════════════════════════════════════════════════════════════════════════
def select_features(df: pd.DataFrame, k: int = 10,
                    name: str = "") -> tuple[pd.DataFrame, list]:
    """
    Not all features are equally useful. Some are noise!
    SelectKBest picks the top-k features using ANOVA F-test.
    
    F-test asks: "Does this feature's value differ between normal and faulty vehicles?"
    High F-score → very informative feature → KEEP
    Low F-score  → not very useful → DROP
    """
    feature_cols = [c for c in df.columns if c != "fault"]
    X = df[feature_cols]
    y = df["fault"]

    selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
    selector.fit(X, y)

    # Get feature names and their scores
    scores = pd.Series(selector.scores_, index=feature_cols).sort_values(
        ascending=False)

    # Keep only selected features
    selected_mask  = selector.get_support()
    selected_feats = [f for f, s in zip(feature_cols, selected_mask) if s]

    print(f"   [{name}] Selected {len(selected_feats)} / {len(feature_cols)} "
          f"features:")
    print(f"      {selected_feats}")

    # Plot feature importance scores
    _plot_feature_scores(scores, name)

    return df[selected_feats + ["fault"]], selected_feats


def _plot_feature_scores(scores: pd.Series, name: str) -> None:
    """Bar chart of F-scores for each feature."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    colors = [PALETTE[0] if i < 10 else PALETTE[1]
              for i in range(len(scores))]
    bars = ax.barh(scores.index[::-1], scores.values[::-1],
                   color=colors[::-1], edgecolor="#333")
    ax.set_xlabel("F-Score (higher = more useful)", color="white")
    ax.set_title(f"📊 [{name}] Feature Importance — SelectKBest F-Scores",
                 color="white", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=scores.iloc[min(10, len(scores))-1],
               color=PALETTE[2], linestyle="--", alpha=0.7,
               label="Selection cutoff")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = f"outputs/feature_scores_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"   📈 Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def preprocess_pipeline(df: pd.DataFrame,
                        name: str,
                        engineer_fn,
                        k_features: int = 10):
    """
    Runs ALL preprocessing steps in order:
    1. Plot raw distributions
    2. Handle missing values
    3. Remove outliers
    4. Engineer new features
    5. Normalize
    6. Select top features
    7. Save cleaned dataset
    """
    safe_name = name.lower().replace(" ", "_")
    print(f"\n{'─'*55}")
    print(f"  🔧 Preprocessing: {name}")
    print(f"{'─'*55}")

    # Step 1: Visualize raw data
    plot_distributions(df, name, f"distributions_{safe_name}.png")
    plot_correlation(df,   name, f"correlation_{safe_name}.png")

    # Step 2: Missing values
    df = handle_missing_values(df, name)

    # Step 3: Outliers
    df = remove_outliers(df, name)

    # Step 4: Feature engineering
    df = engineer_fn(df)

    # Step 5: Normalize
    df, scaler = normalize_features(df)

    # Step 6: Feature selection
    df, selected = select_features(df, k=k_features, name=name)

    # Step 7: Save
    out_path = f"data/{safe_name}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"   ✅ Saved cleaned data → {out_path}  ({df.shape})")

    return df, scaler, selected


if __name__ == "__main__":
    print("\n" + "🔧 " * 20)
    print("  VEHICLE HEALTH MONITORING SYSTEM")
    print("  Step 2: Data Preprocessing")
    print("🔧 " * 20 + "\n")

    # Load
    t_df, b_df = load_data()

    # Class balance plot (before preprocessing)
    print("\n📊 Plotting class balance...")
    plot_class_balance(t_df, b_df)

    # Preprocess both
    t_clean, t_scaler, t_feats = preprocess_pipeline(
        t_df, "Vehicle Telemetry",
        engineer_features_telemetry, k_features=10)

    b_clean, b_scaler, b_feats = preprocess_pipeline(
        b_df, "Battery Health",
        engineer_features_battery,  k_features=8)

    print("\n\n✅ Step 2 Complete! Cleaned data saved to /data folder.")
    print("   Charts saved to /outputs folder.")
    print("   ➡️  Next: Run python step3_train_models.py\n")
