

import numpy as np
import pandas as pd
import os

# ─── Reproducibility ─────────────────────────────────────────────────────────
# Setting a random seed means every time you run this, you get the SAME data.
# Think of it like using the same dice roll pattern each time.
np.random.seed(42)

# ─── Output folder ───────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)   # Creates /data folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# DATASET 1 — Vehicle Telemetry
# Imagine a car sending sensor readings every minute during a trip.
# ════════════════════════════════════════════════════════════════════════════
def generate_telemetry_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Creates a fake but realistic vehicle telemetry dataset.

    Each row = one sensor reading snapshot from a vehicle.
    The TARGET column is 'fault' (0 = normal, 1 = fault detected).

    Sensors included:
      - engine_temp      : Engine temperature in °C
      - rpm              : Engine revolutions per minute
      - vehicle_speed    : Speed in km/h
      - oil_pressure     : Oil pressure in PSI
      - fuel_level       : Fuel remaining (0–100%)
      - coolant_temp     : Coolant temperature in °C
      - throttle_position: Throttle opening (0–100%)
      - brake_pressure   : Brake pressure in bar
      - vibration        : Vibration sensor reading
      - exhaust_temp     : Exhaust gas temperature in °C
    """
    print("📊 Generating Vehicle Telemetry Dataset...")

    # ── Normal operating conditions ──────────────────────────────────────────
    engine_temp      = np.random.normal(90,  10, n_samples)    # avg 90°C ±10
    rpm              = np.random.normal(2500, 500, n_samples)  # avg 2500 RPM
    vehicle_speed    = np.random.uniform(0,  120, n_samples)   # 0–120 km/h
    oil_pressure     = np.random.normal(40,   5, n_samples)    # avg 40 PSI
    fuel_level       = np.random.uniform(5,  100, n_samples)   # 5–100%
    coolant_temp     = np.random.normal(85,   8, n_samples)    # avg 85°C
    throttle_position= np.random.uniform(0,  100, n_samples)   # 0–100%
    brake_pressure   = np.random.normal(30,   5, n_samples)    # avg 30 bar
    vibration        = np.random.normal(0.3, 0.1, n_samples)   # low vibration
    exhaust_temp     = np.random.normal(350, 50, n_samples)    # avg 350°C

    # ── Introduce faults in ~25% of samples ─────────────────────────────────
    # We randomly pick 25% of rows to be "faulty"
    fault_indices = np.random.choice(n_samples,
                                     size=int(0.25 * n_samples),
                                     replace=False)

    # When there's a fault, sensors go out of normal range:
    engine_temp[fault_indices]       += np.random.normal(30, 10,
                                         len(fault_indices))   # overheating
    oil_pressure[fault_indices]      -= np.random.normal(15,  5,
                                         len(fault_indices))   # pressure drop
    vibration[fault_indices]         += np.random.normal(0.5, 0.2,
                                         len(fault_indices))   # more shaking
    coolant_temp[fault_indices]      += np.random.normal(20,  8,
                                         len(fault_indices))   # overheating
    exhaust_temp[fault_indices]      += np.random.normal(100, 30,
                                         len(fault_indices))   # hot exhaust

    # ── Create fault labels ──────────────────────────────────────────────────
    fault_label = np.zeros(n_samples, dtype=int)   # everyone starts as normal
    fault_label[fault_indices] = 1                  # mark faulty rows as 1

    # ── Inject missing values (~3%) to simulate real sensor dropout ──────────
    for col_data in [engine_temp, rpm, oil_pressure, vibration]:
        missing_idx = np.random.choice(n_samples,
                                       size=int(0.03 * n_samples),
                                       replace=False)
        col_data[missing_idx] = np.nan  # NaN = "Not a Number" = missing

    # ── Assemble into a DataFrame (like an Excel table in Python) ────────────
    df = pd.DataFrame({
        "engine_temp":       engine_temp,
        "rpm":               rpm,
        "vehicle_speed":     vehicle_speed,
        "oil_pressure":      oil_pressure,
        "fuel_level":        fuel_level,
        "coolant_temp":      coolant_temp,
        "throttle_position": throttle_position,
        "brake_pressure":    brake_pressure,
        "vibration":         vibration,
        "exhaust_temp":      exhaust_temp,
        "fault":             fault_label          # ← This is what we PREDICT
    })

    df.to_csv("data/vehicle_telemetry.csv", index=False)
    print(f"   ✅ Saved → data/vehicle_telemetry.csv  ({len(df)} rows, "
          f"{df.columns.size} columns)")
    return df


# ════════════════════════════════════════════════════════════════════════════
# DATASET 2 — Battery Health
# Simulates electric vehicle (EV) or hybrid battery sensor readings.
# ════════════════════════════════════════════════════════════════════════════
def generate_battery_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Creates a fake but realistic battery health dataset.

    Sensors included:
      - voltage         : Battery pack voltage (V)
      - current         : Charge/discharge current (A)
      - temperature     : Battery cell temperature (°C)
      - state_of_charge : How full the battery is (0–100%)
      - internal_resist : Internal resistance (Ω) — increases as battery ages
      - cycle_count     : How many charge cycles the battery has gone through
      - cell_imbalance  : Voltage difference between cells (V)
      - power_output    : Current power output (kW)
      - charge_rate     : How fast it's charging (C-rate)
      - age_months      : Battery age in months
    """
    print("🔋 Generating Battery Health Dataset...")

    voltage          = np.random.normal(370,  15, n_samples)    # ~370V pack
    current          = np.random.normal(50,   20, n_samples)    # ~50A
    temperature      = np.random.normal(25,    5, n_samples)    # 25°C nominal
    state_of_charge  = np.random.uniform(10,  100, n_samples)   # 10–100%
    internal_resist  = np.random.normal(0.05, 0.01, n_samples)  # 0.05 Ω
    cycle_count      = np.random.randint(0,  1000, n_samples)   # 0–1000 cycles
    cell_imbalance   = np.random.normal(0.02, 0.005, n_samples) # small imbalance
    power_output     = np.random.normal(50,   10, n_samples)    # ~50 kW
    charge_rate      = np.random.uniform(0.1,  2.0, n_samples)  # 0.1–2C
    age_months       = np.random.randint(0,    84, n_samples)   # 0–7 years

    # ── Fault logic ──────────────────────────────────────────────────────────
    fault_indices = np.random.choice(n_samples,
                                     size=int(0.20 * n_samples),
                                     replace=False)

    # Faulty batteries show these anomalies:
    temperature[fault_indices]     += np.random.normal(20, 8, len(fault_indices))
    internal_resist[fault_indices] += np.random.normal(0.05, 0.02, len(fault_indices))
    cell_imbalance[fault_indices]  += np.random.normal(0.1, 0.03, len(fault_indices))
    voltage[fault_indices]         -= np.random.normal(20, 8, len(fault_indices))

    fault_label = np.zeros(n_samples, dtype=int)
    fault_label[fault_indices] = 1

    # ── Missing values ───────────────────────────────────────────────────────
    for col_data in [voltage, current, temperature, internal_resist]:
        missing_idx = np.random.choice(n_samples,
                                       size=int(0.02 * n_samples),
                                       replace=False)
        col_data[missing_idx] = np.nan

    df = pd.DataFrame({
        "voltage":         voltage,
        "current":         current,
        "temperature":     temperature,
        "state_of_charge": state_of_charge,
        "internal_resist": internal_resist,
        "cycle_count":     cycle_count,
        "cell_imbalance":  cell_imbalance,
        "power_output":    power_output,
        "charge_rate":     charge_rate,
        "age_months":      age_months,
        "fault":           fault_label
    })

    df.to_csv("data/battery_health.csv", index=False)
    print(f"   ✅ Saved → data/battery_health.csv  ({len(df)} rows, "
          f"{df.columns.size} columns)")
    return df


# ════════════════════════════════════════════════════════════════════════════
# EXPLORE the datasets (print summaries to understand the data)
# ════════════════════════════════════════════════════════════════════════════
def explore_dataset(df: pd.DataFrame, name: str) -> None:
    """Prints useful statistics about a dataset so you understand it."""
    print(f"\n{'='*60}")
    print(f"  EXPLORING: {name}")
    print(f"{'='*60}")

    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\n📋 First 3 rows:")
    print(df.head(3).to_string())

    print(f"\n📊 Statistical Summary:")
    print(df.describe().round(2).to_string())

    print(f"\n❓ Missing Values per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "percent %": missing_pct})
    print(missing_df[missing_df["count"] > 0].to_string())

    print(f"\n🎯 Target Column (fault) distribution:")
    counts = df["fault"].value_counts()
    total  = len(df)
    print(f"   Normal (0): {counts.get(0, 0):>5}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"   Fault  (1): {counts.get(1, 0):>5}  ({counts.get(1,0)/total*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN — runs when you execute this file
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "🚗 " * 20)
    print("  VEHICLE HEALTH MONITORING SYSTEM")
    print("  Step 1: Dataset Generation & Exploration")
    print("🚗 " * 20 + "\n")

    # Generate both datasets
    telemetry_df = generate_telemetry_dataset(n_samples=5000)
    battery_df   = generate_battery_dataset(n_samples=3000)

    # Explore them
    explore_dataset(telemetry_df, "Vehicle Telemetry Dataset")
    explore_dataset(battery_df,   "Battery Health Dataset")

    print("\n✅ Step 1 Complete! Data saved to /data folder.")
    print("   ➡️  Next: Run python step2_preprocess.py\n")
