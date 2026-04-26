# RNN Student Performance Evaluator — Final Explanation Report
## Task 11: Conceptual Understanding

---

### 1. What is an RNN in your own words?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to process data **sequentially** — one step at a time. Unlike a regular ANN which reads all inputs at once, an RNN has a **hidden state** (think of it as "memory") that gets updated at each step. After reading step 1, the hidden state carries information forward to step 2, then step 3, and so on. This makes RNN naturally suited for sequences: text, speech, time-series, and in our case, we treat each student feature as one step in a sequence.

The core equation at each time step t:

```
h_t = tanh(W_xh · x_t  +  W_hh · h_{t-1}  +  b_h)
```

Where `h_t` = new hidden state, `x_t` = current input, `h_{t-1}` = previous hidden state.

---

### 2. What function did your model learn?

The RNN learned:

```
f(attendance → assignment → quiz → mid → study_hours)  →  {0=Fail, 1=Pass}
```

It processes features **sequentially** — attendance first, then assignment, etc. — and at each step updates its internal memory. The final hidden state (after reading all 5 features) is used to predict Pass/Fail via a sigmoid output neuron.

The model learned that **strong scores across all steps** lead to a high-confidence Pass, while **weak early signals** (low attendance) bias the hidden state negatively even before later features are read.

---

### 3. How does your system evaluate a new student?

1. **Input**: 5 values (attendance, assignment, quiz, mid, study_hours)
2. **Scale**: StandardScaler normalises them to zero-mean, unit-variance
3. **Reshape**: Values become a sequence of shape `(5, 1)` — 5 time steps
4. **Sequential forward pass**:
   - Step 1: RNN reads `attendance` → updates hidden state h₁
   - Step 2: RNN reads `assignment` → updates to h₂ (using h₁)
   - Step 3: RNN reads `quiz` → updates to h₃
   - Step 4: RNN reads `mid` → updates to h₄
   - Step 5: RNN reads `study_hours` → updates to h₅
5. **Output**: h₅ is passed through sigmoid → probability of Pass
6. **Decision**: prob ≥ 0.5 → Pass, else Fail

---

### 4. Why is scaling important for RNN?

RNN uses **tanh** activation for hidden states and **sigmoid** for output. Both functions **saturate** for large input values:
- tanh(100) ≈ 1.0  (flat region, gradient ≈ 0)
- tanh(-100) ≈ -1.0 (flat region, gradient ≈ 0)

When gradients vanish during **Backpropagation Through Time (BPTT)**, weights stop updating and the model cannot learn. StandardScaler keeps inputs near zero (mean=0, std=1), keeping activations in the **sensitive region** of tanh where gradients are non-zero and meaningful learning occurs.

---

### 5. Limitations of this model

| Limitation | Detail |
|---|---|
| **Not truly sequential data** | Student features aren't naturally a time series. RNN adds complexity without a clear sequential benefit over ANN. |
| **Vanilla RNN — vanishing gradients** | Long sequences cause gradients to shrink. LSTM/GRU would handle this better. |
| **Slow training** | Sample-by-sample BPTT in pure NumPy is slow vs vectorised frameworks. |
| **Synthetic dataset** | 600 artificial samples; real students may behave differently. |
| **No temporal history** | We have only one snapshot per student, not semester-by-semester progression. |
| **Binary output only** | Pass/Fail loses nuance. A grade (0–100) would be more informative. |
| **Fixed architecture** | No hyperparameter tuning (learning rate, hidden size, epochs). |

---

### RNN vs ANN Comparison

| Aspect | ANN (MLPClassifier) | RNN (Elman, scratch) |
|---|---|---|
| Input style | All features at once | One feature per time step |
| Memory | None | Hidden state h_t |
| Test Accuracy | **89.17%** | **85.83%** |
| Training speed | Fast | Slower (BPTT) |
| Best for | Tabular data | Sequences, time-series |
| Complexity | Lower | Higher |

**Conclusion**: For this tabular dataset, ANN performs slightly better. RNN shines when data has genuine temporal/sequential structure (e.g., weekly quiz scores over a semester).

---

*Report generated as part of RNN-Based Student Performance Evaluator assignment.*
