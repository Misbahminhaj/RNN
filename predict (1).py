"""
predict.py
==========
Task 7 – Reusable evaluation function using saved RNN model.
Run directly for command-line demo.
"""

import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_rnn    = joblib.load(os.path.join(BASE_DIR, "rnn_model.joblib"))
_scaler = joblib.load(os.path.join(BASE_DIR, "rnn_scaler.joblib"))


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict whether a student will Pass or Fail using trained RNN.

    Parameters
    ----------
    attendance  : float  attendance percentage (0–100)
    assignment  : float  assignment score (0–100)
    quiz        : float  quiz score (0–100)
    mid         : float  mid-term score (0–100)
    study_hours : float  weekly study hours (0–20)

    Returns
    -------
    dict:
        prediction  – 0 (Fail) or 1 (Pass)
        label       – "Pass ✅" or "Fail ❌"
        prob_pass   – Pass probability (%)
        prob_fail   – Fail probability (%)
        performance – "High 🌟" / "Medium ⚠️" / "Low 🔴"
    """
    raw    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = _scaler.transform(raw)          # (1, 5)
    seq    = scaled.reshape(1, 5, 1)         # (1, T=5, feat=1)

    preds, probas = _rnn.predict(seq)
    pred      = int(preds[0])
    prob_pass = float(probas[0]) * 100

    if prob_pass >= 80:
        performance = "High 🌟"
    elif prob_pass >= 50:
        performance = "Medium ⚠️"
    else:
        performance = "Low 🔴"

    return {
        "prediction" : pred,
        "label"      : "Pass ✅" if pred == 1 else "Fail ❌",
        "prob_pass"  : round(prob_pass, 1),
        "prob_fail"  : round(100 - prob_pass, 1),
        "performance": performance,
    }


# ── Command-line demo ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  RNN Student Performance Evaluator (CLI)")
    print("=" * 50)

    try:
        attendance  = float(input("Attendance  (0–100): "))
        assignment  = float(input("Assignment  (0–100): "))
        quiz        = float(input("Quiz        (0–100): "))
        mid         = float(input("Mid-term    (0–100): "))
        study_hours = float(input("Study hours/week   : "))
    except ValueError:
        print("❌ Please enter numeric values.")
        exit(1)

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    print("\n─── RNN Prediction ───────────────────────────")
    print(f"  Result      : {result['label']}")
    print(f"  Performance : {result['performance']}")
    print(f"  Pass prob   : {result['prob_pass']}%")
    print(f"  Fail prob   : {result['prob_fail']}%")
    print("─────────────────────────────────────────────")
