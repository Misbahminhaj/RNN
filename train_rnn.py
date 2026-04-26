"""
train_rnn.py
============
RNN-Based Student Performance Evaluator
Tasks 1 → 8: Load, preprocess, build RNN (from scratch with NumPy),
train, evaluate, save.

NOTE ON RNN FOR TABULAR DATA:
  True RNNs are designed for sequential/time-series data (text, speech).
  Here we treat each student's 5 features as a TIME SEQUENCE of length 5
  (each feature = one time step). This is a valid educational demonstration
  of how RNN processes inputs step-by-step vs ANN all-at-once.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# TASK 1 – Understand the Dataset
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  TASK 1 – Dataset Overview")
print("=" * 65)

df = pd.read_excel("dataset.xlsx")

print("\n📋 First 5 rows:")
print(df.head())
print(f"\n📐 Shape: {df.shape}  ({df.shape[0]} students, {df.shape[1]} columns)")
print(f"\n🔤 Column names: {df.columns.tolist()}")

print("""
📖 Column Descriptions:
  attendance  – % of classes attended         (0–100)
  assignment  – assignment marks               (0–100)
  quiz        – quiz marks                    (0–100)
  mid         – mid-term exam marks           (0–100)
  study_hours – self-study hours per week     (0–20)
  result      – 0 = Fail, 1 = Pass  ← TARGET

🧩 Problem Type : CLASSIFICATION
   Reason       : target is binary (0/1), not continuous.

🔄 RNN Interpretation:
   We treat the 5 features as a SEQUENCE of 5 time steps.
   Step 1 → attendance
   Step 2 → assignment
   Step 3 → quiz
   Step 4 → mid
   Step 5 → study_hours
   The RNN reads them one-by-one, updating its hidden state each step,
   then predicts Pass/Fail from the final hidden state.
""")
print("Class distribution:")
print(df["result"].value_counts())

# ─────────────────────────────────────────────────────────────
# TASK 3 – Data Preprocessing
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TASK 3 – Preprocessing")
print("=" * 65)

FEATURES = ["attendance", "assignment", "quiz", "mid", "study_hours"]
TARGET   = "result"

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Reshape for RNN: (samples, time_steps, features_per_step)
# Each feature = 1 time step, feature_dim = 1
X_train_rnn = X_train_s.reshape(X_train_s.shape[0], 5, 1)
X_test_rnn  = X_test_s.reshape(X_test_s.shape[0],  5, 1)

print(f"Training samples : {X_train_rnn.shape[0]}")
print(f"Testing  samples : {X_test_rnn.shape[0]}")
print(f"RNN input shape  : {X_train_rnn.shape}  (samples, timesteps, features)")
print("""
🔍 Why scaling in RNN?
   RNN hidden states are computed via tanh/sigmoid. These functions
   saturate (output ≈ ±1) for large inputs, killing gradients.
   Scaling keeps inputs in a range where activations are sensitive,
   allowing gradients to flow backward and weights to update properly.
""")

# ─────────────────────────────────────────────────────────────
# TASK 4 – Build RNN from Scratch (NumPy)
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  TASK 4 – RNN Architecture (built from scratch)")
print("=" * 65)

print("""
🧠 Architecture:
   Input  → sequence of 5 time steps (1 feature each)
   RNN    → hidden size = 16 (tanh activation)
   Dense  → 16 → 1 neuron (sigmoid → binary output)

📚 Key Concepts:
   Neuron         – computes weighted sum + bias + activation
   Activation fn  – tanh: squashes output to (-1, 1)
                    sigmoid: squashes to (0, 1) for probability
   Hidden layer   – RNN hidden state = "memory" of past steps
   Hidden state   – h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
                    carries info from step to step

🔄 RNN vs ANN:
   ANN: all features fed at once → parallel
   RNN: features fed one-by-one → sequential with memory
""")

class SimpleRNN:
    """
    Minimal RNN classifier built with pure NumPy.
    Forward pass: Elman RNN + sigmoid output.
    Training: BPTT (Backprop Through Time) + mini-batch SGD.
    """

    def __init__(self, input_size=1, hidden_size=16, output_size=1, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr

        # Xavier initialisation
        s = lambda r, c: np.random.randn(r, c) * np.sqrt(2.0 / (r + c))
        self.W_xh = s(hidden_size, input_size)   # input  → hidden
        self.W_hh = s(hidden_size, hidden_size)  # hidden → hidden
        self.b_h  = np.zeros((hidden_size, 1))

        self.W_hy = s(output_size, hidden_size)  # hidden → output
        self.b_y  = np.zeros((output_size, 1))

    # ── Activations ──────────────────────────────────────────
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    @staticmethod
    def sigmoid_deriv(s):
        return s * (1 - s)

    @staticmethod
    def tanh_deriv(t):
        return 1 - t ** 2

    # ── Forward pass for one sample ──────────────────────────
    def forward(self, x_seq):
        """
        x_seq: (T, input_size)
        Returns: output probability, list of hidden states
        """
        T = x_seq.shape[0]
        h = np.zeros((self.hidden_size, 1))
        hs = [h]
        xs = []

        for t in range(T):
            x_t = x_seq[t].reshape(-1, 1)
            xs.append(x_t)
            h = np.tanh(self.W_xh @ x_t + self.W_hh @ h + self.b_h)
            hs.append(h)

        y_hat = self.sigmoid(self.W_hy @ h + self.b_y)
        return y_hat, hs, xs

    # ── BPTT for one sample ──────────────────────────────────
    def backward(self, x_seq, y_true, y_hat, hs, xs):
        T = x_seq.shape[0]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)

        # Output layer gradient
        dy = y_hat - y_true                       # BCE gradient
        dW_hy += dy * hs[-1].T
        db_y  += dy

        # Backprop through time
        dh_next = self.W_hy.T @ dy
        for t in reversed(range(T)):
            dh = dh_next * self.tanh_deriv(hs[t + 1])
            dW_xh += dh @ xs[t].T
            dW_hh += dh @ hs[t].T
            db_h  += dh
            dh_next = self.W_hh.T @ dh

        # Gradient clipping (prevents exploding gradients)
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -1, 1, out=grad)

        # SGD update
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.b_h  -= self.lr * db_h
        self.W_hy -= self.lr * dW_hy
        self.b_y  -= self.lr * db_y

    # ── Training loop ────────────────────────────────────────
    def train(self, X, y, epochs=100, verbose=True):
        loss_history = []
        acc_history  = []
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            total_loss = 0
            correct    = 0

            for i in range(n):
                x_seq  = X[i]                     # (T, 1)
                y_true = float(y[i])

                y_hat, hs, xs = self.forward(x_seq)
                p = float(y_hat.flatten()[0])

                # Binary cross-entropy
                loss = -(y_true * np.log(p + 1e-8)
                         + (1 - y_true) * np.log(1 - p + 1e-8))
                total_loss += loss

                self.backward(x_seq, y_true, y_hat, hs, xs)

                if (p >= 0.5) == bool(y_true):
                    correct += 1

            avg_loss = total_loss / n
            acc      = correct / n
            loss_history.append(avg_loss)
            acc_history.append(acc)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  Epoch {epoch:>3}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc*100:.1f}%")

        return loss_history, acc_history

    # ── Predict ──────────────────────────────────────────────
    def predict(self, X):
        preds  = []
        probas = []
        for i in range(X.shape[0]):
            y_hat, _, _ = self.forward(X[i])
            p = float(y_hat.flatten()[0])
            probas.append(p)
            preds.append(1 if p >= 0.5 else 0)
        return np.array(preds), np.array(probas)

# ─────────────────────────────────────────────────────────────
# TASK 5 – Train
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TASK 5 – Training")
print("=" * 65)

rnn = SimpleRNN(input_size=1, hidden_size=16, output_size=1, lr=0.005)
print("\n🚀 Starting RNN training (100 epochs, BPTT + SGD)...\n")
loss_hist, acc_hist = rnn.train(X_train_rnn, y_train, epochs=100, verbose=True)
print("\n✅ Training complete!")

# ─────────────────────────────────────────────────────────────
# TASK 6 – Evaluate
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TASK 6 – Evaluation on Test Set")
print("=" * 65)

y_pred, y_proba = rnn.predict(X_test_rnn)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
cr  = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])

print(f"\n🎯 Accuracy : {acc * 100:.2f}%")
print(f"\n🗂  Confusion Matrix:\n{cm}")
print(f"\n📊 Classification Report:\n{cr}")

print(f"""
💬 Interpretation:
   Accuracy = {acc*100:.1f}% → model correctly predicts {acc*100:.1f}/100 students.
   Confusion Matrix (rows=Actual, cols=Predicted):
     TN={cm[0,0]}  FP={cm[0,1]}   ← actually Fail
     FN={cm[1,0]}  TP={cm[1,1]}   ← actually Pass

   FP = predicted Pass but actually Fail (false hope)
   FN = predicted Fail but actually Pass (missed talent)
""")

# ── Plots ─────────────────────────────────────────────────────
# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("RNN – Confusion Matrix")
plt.tight_layout()
plt.savefig("rnn_confusion_matrix.png", dpi=150)
print("📈 Confusion matrix saved → rnn_confusion_matrix.png")

# Training curves
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.plot(loss_hist, color="purple"); ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax2.plot(acc_hist,  color="green");  ax2.set_title("Training Accuracy")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig("rnn_training_curve.png", dpi=150)
print("📈 Training curve saved → rnn_training_curve.png")

# ─────────────────────────────────────────────────────────────
# TASK 7 – Evaluation Function
# ─────────────────────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict student result using trained RNN.

    Parameters
    ----------
    attendance, assignment, quiz, mid : float  (0–100)
    study_hours                        : float  (0–20)

    Returns
    -------
    dict with prediction, label, probabilities, performance tier
    """
    raw = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(raw)                  # (1, 5)
    seq = scaled.reshape(1, 5, 1)                   # (1, T, 1)

    pred_arr, proba_arr = rnn.predict(seq)
    pred = int(pred_arr[0])
    prob_pass = float(proba_arr[0]) * 100

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

# Quick tests
print("\n" + "=" * 65)
print("  TASK 7 – evaluate_student() Tests")
print("=" * 65)
tests = [
    (90, 95, 88, 80, 15, "Strong student"),
    (30, 40, 25, 30,  2, "Weak student"),
    (65, 60, 55, 50,  6, "Average student"),
]
for att, asn, qz, md, sh, label in tests:
    r = evaluate_student(att, asn, qz, md, sh)
    print(f"  {label:20s} → {r['label']} ({r['prob_pass']}%) {r['performance']}")

# ─────────────────────────────────────────────────────────────
# TASK 8 – Save Model
# ─────────────────────────────────────────────────────────────
joblib.dump(rnn,    "rnn_model.joblib")
joblib.dump(scaler, "rnn_scaler.joblib")
print("\n💾 rnn_model.joblib  saved")
print("💾 rnn_scaler.joblib saved")
print("""
📝 Why save both?
   rnn_model.joblib  – contains all learned weights (W_xh, W_hh, W_hy, biases)
   rnn_scaler.joblib – contains mean & std from training data
   Without scaler: raw inputs won't match training scale → garbage predictions.
   Always load & use them together.
""")

print("=" * 65)
print("✅ RNN Training Pipeline Complete!")
print("=" * 65)
