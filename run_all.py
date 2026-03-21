"""
============================================================
 VEHICLE HEALTH MONITORING SYSTEM
 run_all.py — Master Script (Run Everything at Once)
============================================================

WHAT THIS DOES:
  Runs all 4 steps in sequence:
    Step 1 → Generate datasets
    Step 2 → Preprocess & clean data
    Step 3 → Train Random Forest, XGBoost, LSTM
    Step 4 → Evaluate, compare, SHAP, predict

HOW TO RUN:
  In VS Code terminal: python run_all.py
  In Jupyter:          !python run_all.py
  In Google Colab:     !python run_all.py

ESTIMATED TIME: ~2–5 minutes depending on your computer.
"""

import subprocess
import sys
import time

STEPS = [
    ("step1_generate_data.py",  "Step 1: Dataset Generation"),
    ("step2_preprocess.py",     "Step 2: Data Preprocessing"),
    ("step3_train_models.py",   "Step 3: Model Training"),
    ("step4_evaluate_compare.py","Step 4: Evaluation & Comparison"),
]

def run_step(script: str, label: str) -> bool:
    """Runs a Python script and prints its output live."""
    print(f"\n{'━' * 65}")
    print(f"  🚀 Running: {label}")
    print(f"{'━' * 65}\n")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # Show output live in terminal
        text=True
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  ✅ {label} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ {label} FAILED after {elapsed:.1f}s")
        print("  Check the error message above and fix it before continuing.")
        return False


if __name__ == "__main__":
    print("\n" + "🚗" * 32)
    print("   VEHICLE HEALTH MONITORING SYSTEM")
    print("   Full Pipeline Run")
    print("🚗" * 32)

    overall_start = time.time()
    all_passed    = True

    for script, label in STEPS:
        ok = run_step(script, label)
        if not ok:
            all_passed = False
            print(f"\n⛔ Pipeline stopped at: {label}")
            print("   Fix the error above and re-run: python run_all.py\n")
            sys.exit(1)

    total_time = time.time() - overall_start

    print("\n" + "🎉" * 32)
    print("  ALL STEPS COMPLETE! 🏆")
    print(f"  Total time: {total_time:.1f} seconds")
    print("🎉" * 32)
    print("\n  📁 Check these folders for output:")
    print("     /outputs  — All charts and plots")
    print("     /models   — Saved trained models")
    print("     /data     — Clean datasets\n")
