"""
app.py
======
Task 9 – Streamlit UI for RNN Student Performance Evaluator.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from predict import evaluate_student

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RNN Student Evaluator",
    page_icon="🔄",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────
st.title("🔄 RNN Student Performance Evaluator")
st.markdown(
    "Enter a student's details. The **Recurrent Neural Network (RNN)** "
    "reads each feature as a sequential time step and predicts "
    "**Pass** or **Fail**."
)

with st.expander("ℹ️ How does the RNN work here?"):
    st.markdown("""
    Unlike ANN which reads all features at once, **RNN reads them one by one**:

    | Time Step | Feature |
    |---|---|
    | Step 1 | Attendance |
    | Step 2 | Assignment |
    | Step 3 | Quiz |
    | Step 4 | Mid-term |
    | Step 5 | Study Hours |

    At each step, RNN updates its **hidden state** (memory), then predicts from the final state.
    """)

st.divider()

# ── Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    attendance  = st.slider("📅 Attendance (%)",    0, 100, 75)
    assignment  = st.slider("📝 Assignment Score",  0, 100, 70)
    quiz        = st.slider("🧩 Quiz Score",        0, 100, 65)

with col2:
    mid         = st.slider("📖 Mid-term Score",    0, 100, 60)
    study_hours = st.slider("⏱️ Study Hours / Week", 0,  20,  6)

st.divider()

# ── RNN Steps Visualization ───────────────────────────────────
st.markdown("#### 🔄 RNN Sequential Processing")
features_display = {
    "Attendance": attendance,
    "Assignment": assignment,
    "Quiz": quiz,
    "Mid-term": mid,
    "Study Hrs": study_hours * 5,
}

fig_seq, ax = plt.subplots(figsize=(8, 2))
colors = ["#7C3AED", "#9333EA", "#A855F7", "#C084FC", "#DDD6FE"]
for i, (name, val) in enumerate(features_display.items()):
    ax.barh(0, 1, left=i, color=colors[i], edgecolor="white", height=0.5)
    ax.text(i + 0.5, 0, f"t{i+1}\n{name}\n{val:.0f}",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
ax.set_title("RNN Time Steps (left → right)", fontsize=10)
st.pyplot(fig_seq)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔮 Run RNN Prediction", use_container_width=True, type="primary"):

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    # Result banner
    if result["prediction"] == 1:
        st.success(f"### {result['label']}  —  {result['performance']}")
    else:
        st.error(f"### {result['label']}  —  {result['performance']}")

    # Probability display
    st.markdown("#### 📊 Prediction Confidence")
    c1, c2 = st.columns(2)
    c1.metric("✅ Pass Probability", f"{result['prob_pass']}%")
    c2.metric("❌ Fail Probability", f"{result['prob_fail']}%")

    st.markdown("**Pass**")
    st.progress(int(result["prob_pass"]))
    st.markdown("**Fail**")
    st.progress(int(result["prob_fail"]))

    st.divider()

    # Profile bar chart vs average
    st.markdown("#### 📈 Student Profile vs Class Average")
    labels   = ["Attend.", "Assign.", "Quiz", "Mid", "Study×5"]
    student  = [attendance, assignment, quiz, mid, study_hours * 5]
    avg_vals = [73, 68, 60, 57, 35]

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, student,  0.35, label="This Student", color="#7C3AED")
    ax2.bar(x + 0.2, avg_vals, 0.35, label="Class Avg",    color="#F59E0B", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 110); ax2.legend()
    ax2.set_title("Student vs Average")
    st.pyplot(fig2)

    st.divider()

    # Interpretation
    st.markdown("#### 💡 RNN Recommendation")
    if "High" in result["performance"]:
        st.info("RNN detected **strong sequential signals** across all time steps. "
                "Student shows consistent performance. Keep it up!")
    elif "Medium" in result["performance"]:
        st.warning("RNN detected **mixed signals**. Focus on improving quiz scores "
                   "and increasing study hours for a stronger Pass.")
    else:
        st.error("RNN detected **weak signals** throughout the sequence. "
                 "Immediate improvement in attendance and study habits required.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This RNN")
    st.markdown("""
**Model:** Simple RNN (NumPy from scratch)

**Architecture:**
- Input: 5 time steps (1 feature/step)
- Hidden: 16 neurons (tanh)
- Output: 1 neuron (sigmoid)
- Training: BPTT + SGD

**Dataset:** 600 synthetic students

**Test Accuracy:** 85.83%

**Key equation:**
`h_t = tanh(W·x_t + U·h_{t-1} + b)`

**Performance Levels:**
- 🌟 High  → Pass ≥ 80%
- ⚠️ Medium → Pass 50–80%
- 🔴 Low   → Pass < 50%
    """)

    st.divider()
    st.markdown("**RNN vs ANN**")
    st.markdown("🔄 RNN: reads features **sequentially**")
    st.markdown("⚡ ANN: reads features **all at once**")
