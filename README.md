# 🚗 Vehicle Health Monitoring System
### Using Machine Learning for Predictive Maintenance

---

## 📋 What This Project Does

This project builds a complete **AI-powered system** that:
1. Reads vehicle sensor data (engine temperature, RPM, oil pressure, etc.)
2. Learns the difference between **normal** and **faulty** vehicle conditions
3. Predicts whether a vehicle is about to break down **before it happens**
4. Explains **why** it made each prediction (Explainable AI)

---

## 🗂️ Project Structure

```
vehicle_health_monitor/
│
├── 📄 run_all.py                ← Run this to execute everything
│
├── 📄 step1_generate_data.py    ← Creates synthetic datasets
├── 📄 step2_preprocess.py       ← Cleans and prepares data
├── 📄 step3_train_models.py     ← Trains 3 ML models
├── 📄 step4_evaluate_compare.py ← Compares models, SHAP, predictions
│
├── 📄 requirements.txt          ← List of packages to install
│
├── 📁 data/                     ← Generated CSV files (auto-created)
├── 📁 outputs/                  ← All charts and graphs (auto-created)
└── 📁 models/                   ← Saved trained models (auto-created)
```

---

## 🚀 Quick Start (Step-by-Step for Absolute Beginners)

### Step 1 — Install Python
Download Python 3.10+ from: https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation!

### Step 2 — Open VS Code
1. Download VS Code: https://code.visualstudio.com/
2. Install the **Python extension** from the Extensions panel (Ctrl+Shift+X)

### Step 3 — Open This Project
1. In VS Code: File → Open Folder → select `vehicle_health_monitor`
2. Open the **Terminal** (Ctrl+` or View → Terminal)

### Step 4 — Install Required Packages
```bash
pip install -r requirements.txt
```
This installs: numpy, pandas, scikit-learn, xgboost, tensorflow, shap, matplotlib, seaborn

### Step 5 — Run the Full Pipeline
```bash
python run_all.py
```
OR run each step individually:
```bash
python step1_generate_data.py
python step2_preprocess.py
python step3_train_models.py
python step4_evaluate_compare.py
```

---

## 📦 Datasets Used

### Dataset 1: Vehicle Telemetry (5,000 samples)
| Feature | Description | Normal Range |
|---------|-------------|--------------|
| engine_temp | Engine temperature | 80–105°C |
| rpm | Engine revolutions per minute | 800–4000 |
| vehicle_speed | Current speed | 0–120 km/h |
| oil_pressure | Oil pressure | 35–55 PSI |
| fuel_level | Remaining fuel | 0–100% |
| coolant_temp | Coolant temperature | 75–95°C |
| throttle_position | Throttle opening | 0–100% |
| brake_pressure | Brake pressure | 20–45 bar |
| vibration | Vibration sensor | 0.1–0.5 |
| exhaust_temp | Exhaust gas temperature | 250–450°C |

### Dataset 2: Battery Health (3,000 samples)
| Feature | Description |
|---------|-------------|
| voltage | Battery pack voltage (V) |
| current | Charge/discharge current (A) |
| temperature | Battery cell temperature (°C) |
| state_of_charge | Battery level (%) |
| internal_resist | Internal resistance (Ω) |
| cycle_count | Charge cycle count |
| cell_imbalance | Cell voltage imbalance |
| age_months | Battery age in months |

---

## 🤖 Machine Learning Models

### 1. 🌲 Random Forest
- **What it is:** A "forest" of 200 decision trees, each votes on the prediction
- **Think of it as:** Asking 200 experts, taking a majority vote
- **Best for:** Tabular sensor data, easy to interpret
- **Key hyperparameters:**
  - `n_estimators=200` — number of trees
  - `max_depth=15` — how deep each tree can grow

### 2. ⚡ XGBoost (eXtreme Gradient Boosting)
- **What it is:** Trees built sequentially, each correcting the last one's mistakes
- **Think of it as:** A student who keeps studying their wrong answers
- **Best for:** Often wins ML competitions, very accurate
- **Key hyperparameters:**
  - `n_estimators=300` — number of boosting rounds
  - `learning_rate=0.05` — size of each correction step

### 3. 🧠 LSTM (Long Short-Term Memory)
- **What it is:** A deep learning model that remembers past sensor readings
- **Think of it as:** A doctor who considers your health history, not just today
- **Best for:** Time-series data where patterns over time matter
- **Architecture:** LSTM(128) → Dropout → LSTM(64) → Dense(32) → Output

---

## 📊 Evaluation Metrics Explained

| Metric | What it means | Target |
|--------|---------------|--------|
| **Accuracy** | % of ALL predictions correct | > 90% |
| **Precision** | Of "fault" predictions, % actually faulty | > 85% |
| **Recall** | Of real faults, % we caught | > 90% ⚠️ |
| **F1 Score** | Balance of Precision & Recall | > 88% |
| **AUC-ROC** | Overall model discrimination ability | > 0.95 |

> ⚠️ **Recall is most important for safety!**  
> A missed fault (False Negative) is more dangerous than a false alarm (False Positive).

---

## 🔍 Explainable AI — SHAP Values

SHAP tells you **WHY** the model flagged a vehicle as faulty:

```
Vehicle X predicted: FAULT (probability = 87%)

SHAP Explanation:
  engine_temp    +0.42  ← Huge contribution (temp too high!)
  oil_pressure   +0.28  ← Oil pressure low
  vibration      +0.18  ← Unusual vibration
  coolant_temp   +0.12  ← Running hot
  fuel_level     -0.03  ← Slight negative (doesn't indicate fault)
```

This is crucial for real-world deployment — mechanics need to know **what to fix!**

---

## 📁 Output Files

After running, check the `/outputs` folder:

| File | Description |
|------|-------------|
| `distributions_vehicle_telemetry.png` | Histogram of all telemetry features |
| `distributions_battery_health.png` | Histogram of all battery features |
| `correlation_vehicle_telemetry.png` | Feature correlation heatmap |
| `class_balance.png` | Normal vs fault distribution |
| `feature_scores_*.png` | Feature selection scores |
| `confusion_matrices.png` | True/false positive/negative breakdown |
| `roc_curves.png` | Model performance curves (AUC) |
| `model_comparison_table.png` | Side-by-side metrics table |
| `model_comparison_bars.png` | Bar chart comparing all models |
| `feature_importance_*.png` | Which features matter most |
| `probability_distribution.png` | Fault probability spread |
| `shap_summary.png` | SHAP beeswarm plot |
| `shap_importance.png` | SHAP feature importance |
| `lstm_training_history.png` | LSTM loss/accuracy curves |

---

## 🔮 Making Predictions

After training, use the predict function directly:

```python
import joblib
import numpy as np
import pandas as pd

# Load the best model
model = joblib.load("models/random_forest.pkl")

# Load feature names
feature_cols = pd.read_csv("data/feature_cols.csv").iloc[:, 0].tolist()

# Your sensor readings
sensor_data = {
    "engine_temp":       135.0,   # ⚠️ Too hot!
    "rpm":               3200.0,
    "vehicle_speed":     65.0,
    "oil_pressure":      22.5,    # ⚠️ Too low!
    "fuel_level":        45.0,
    "coolant_temp":      115.0,
    "throttle_position": 55.0,
    "brake_pressure":    28.0,
    "vibration":         0.95,    # ⚠️ High vibration!
    "exhaust_temp":      480.0,
}

# Predict
X = np.array([sensor_data.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
prediction   = model.predict(X)[0]
fault_prob   = model.predict_proba(X)[0][1] * 100

print(f"Prediction : {'FAULT' if prediction == 1 else 'NORMAL'}")
print(f"Fault Prob : {fault_prob:.1f}%")
```

---

## 🌐 Running on Google Colab

```python
# Upload files to Colab, then run:
!pip install -r requirements.txt
!python run_all.py
```

Or just copy-paste each script into separate Colab cells.

---

## 🆘 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: xgboost` | Run: `pip install xgboost` |
| `ModuleNotFoundError: tensorflow` | Run: `pip install tensorflow` |
| `ModuleNotFoundError: shap` | Run: `pip install shap` |
| `FileNotFoundError: data/vehicle_telemetry_clean.csv` | Run step1 and step2 first |
| `Python not found` | Add Python to PATH or use full path |

---

## 📚 Concepts Glossary

| Term | Plain English Explanation |
|------|---------------------------|
| **Feature** | An input variable (e.g., engine temperature) |
| **Label/Target** | What we're predicting (fault = 0 or 1) |
| **Training data** | Data the model learns from (80%) |
| **Test data** | Data used to evaluate the model (20%) |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Normalization** | Rescaling features to similar ranges |
| **Imputation** | Filling in missing values |
| **Epoch** | One full pass through training data (for LSTM) |
| **SHAP** | Method to explain individual predictions |
| **AUC-ROC** | Area under ROC curve (1.0 = perfect, 0.5 = random) |
| **F1 Score** | Harmonic mean of precision and recall |
| **Cross-validation** | Testing on multiple train/test splits for reliability |

---

*Built with ❤️ using Python, scikit-learn, XGBoost, TensorFlow & SHAP*
