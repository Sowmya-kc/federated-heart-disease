"""app.py -- web interface for the federated heart-disease model.   Run:  streamlit run app.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from predict import HeartRiskModel

st.set_page_config(page_title="Federated Heart Disease Prediction", page_icon="🫀", layout="wide")
MODEL, LOG = "models/global_model.npz", "models/training_log.json"
NAMES = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Blood pressure", "chol": "Cholesterol",
         "fbs": "Fasting blood sugar", "restecg": "Resting ECG", "thalach": "Max heart rate",
         "exang": "Exercise angina", "oldpeak": "ST depression"}

st.title("🫀 Privacy-Preserving Federated Heart Disease Prediction")
st.caption("No patient record was ever shared.")
st.info("""
### Federated Learning Model

**Architecture**

Input features  
↓  
Dense Layer (16 neurons)  
↓  
ReLU activation  
↓  
Dense Layer (8 neurons)  
↓  
ReLU activation  
↓  
Sigmoid output (heart disease probability)

**Training method**

- Algorithm: Federated Averaging (FedAvg)
- Hospitals: Cleveland, Hungarian, VA Long Beach
- Patient records: Never shared
- Only model updates are exchanged
""")

if not (os.path.exists(MODEL) and os.path.exists(LOG)):
    st.error("No trained model found. Run  `python run_federation.py`  first, then reload this page.")
    st.stop()


@st.cache_resource
def load_model():
    return HeartRiskModel(MODEL)


model, log = load_model(), json.load(open(LOG))
tab1, tab2, tab3 = st.tabs(["🩺 Predict for a patient", "📈 How the model was trained", "ℹ️ About & limitations"])

# ------------------------------------------------------------------ tab 1
with tab1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Patient details")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, 55)
        sex = c2.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "No chest pain (asymptomatic)"], index=3)
        c3, c4 = st.columns(2)
        trestbps = c3.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = c4.number_input("Cholesterol (mg/dl)", 80, 600, 240)
        c5, c6 = st.columns(2)
        thalach = c5.number_input("Maximum heart rate achieved", 60, 220, 150)
        oldpeak = c6.number_input("ST depression after exercise", -2.0, 7.0, 1.0, step=0.1)
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        c7, c8 = st.columns(2)
        fbs = c7.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
        exang = c8.selectbox("Exercise-induced angina", ["No", "Yes"])

    patient = dict(age=age, sex=1 if sex == "Male" else 0,
                   cp=["Typical angina", "Atypical angina", "Non-anginal pain", "No chest pain (asymptomatic)"].index(cp) + 1,
                   trestbps=trestbps, chol=chol, fbs=int(fbs == "Yes"),
                   restecg=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
                   thalach=thalach, exang=int(exang == "Yes"), oldpeak=oldpeak)
    risk = model.predict_proba(patient)

    with right:
        st.subheader("Model estimate")
        st.metric("Estimated probability of heart disease", f"{risk:.0%}")
        st.progress(min(max(risk, 0.0), 1.0))
        warnings = model.coverage_warnings(patient)
        if warnings:
            st.info("No low / moderate / high verdict is given: this patient is unlike the training patients, "
                    "so the number above is for reference only.")
        elif risk >= 0.6:
            st.warning("The model estimates a **high** likelihood of heart disease for this input.")
        elif risk >= 0.3:
            st.info("The model estimates a **moderate** likelihood for this input.")
        else:
            st.success("The model estimates a **low** likelihood for this input.")
        st.caption("Bands (low / moderate / high) are illustrative only, not clinical thresholds.")
        label = {"age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
                 "thalach": "Max heart rate", "oldpeak": "ST depression"}
        for c, v, lo, hi in warnings:
            st.warning(f"⚠️ **{label[c]} = {v:g} is unusual for the training patients** (typical range about "
                       f"{max(lo, 0):.0f}–{hi:.0f}). The model is extrapolating here, so treat this estimate as unreliable.")

        st.markdown("**What pushed this estimate up or down?**")
        names = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Blood pressure", "chol": "Cholesterol",
                 "fbs": "Fasting blood sugar", "restecg": "Resting ECG", "thalach": "Max heart rate",
                 "exang": "Exercise angina", "oldpeak": "ST depression"}
        ex = pd.Series(model.explain(patient)).rename(index=names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(ex.index, ex.values * 100, color=["#c0392b" if v > 0 else "#2e86c1" for v in ex.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Change in risk (percentage points) vs. a typical value of that input")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.error("⚠️ Educational research prototype trained on small public datasets. "
             "It is **not** a medical device and must not be used for diagnosis or treatment decisions.")

# ------------------------------------------------------------------ tab 2
with tab2:
    p = log["pooled_test"]
    st.subheader("Final federated evaluation")
    st.caption(f"Each hospital tested the final global model on its own held-out patients ({p['n']} in total) and reported "
               "only counts and scores to the server.")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{p['accuracy']:.3f}"); b.metric("Precision", f"{p['precision']:.3f}")
    c.metric("Recall", f"{p['recall']:.3f}"); d.metric("F1-score", f"{p['f1']:.3f}")
    e.metric("Network traffic", f"{log['traffic_MB_total']:.2f} MB")

    st.subheader("Per-hospital results")
    pc = pd.DataFrame(log["per_client_test"]).T.rename(columns={"n": "test patients", "acc": "accuracy", "prec": "precision",
                                                                "rec": "recall", "f1": "F1", "auc": "AUC"})
    pc.index = [i.replace("_", " ") for i in pc.index]
    st.dataframe(pc.drop(columns="loss").round(3), width="stretch")
    st.caption(f"Read with care: each hospital's test set is small ({int(pc['test patients'].min())}–{int(pc['test patients'].max())} "
               "patients, so one patient is worth about 2 points of accuracy) and this is a single train/test split; "
               "the 10-split study in `results/` is more stable. In the public VA Long Beach data about 75 % of patients have "
               "heart disease, so answering 'disease' for everyone would already score about 0.75 accuracy; "
               f"look at AUC instead. The lowest AUC here is {pc['AUC'].idxmin()} ({pc['AUC'].min():.2f}; 0.5 = random guessing).")

    st.subheader("Training progress (federated validation loss per round)")
    vh = pd.Series(log["val_loss_history"], name="validation loss")
    vh.index.name = "communication round"
    st.line_chart(vh)
    st.subheader("Accuracy improvement during training")

round_history = pd.DataFrame(log["rounds"])

if "accuracy" in round_history.columns:
    st.line_chart(
        round_history.set_index("round")["accuracy"]
    )
else:
    st.caption(
        "Accuracy history was not stored during this training run."
    )
    st.caption(f"The server kept the global model from round {log['best_round']} (lowest validation loss). "
               "Validation loss is computed by the hospitals on their own data.")

    st.subheader("Which inputs does the federated model rely on?")
    imp = pd.Series(log["importance"]).rename(index=NAMES).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 3.6))
    ax2.barh(imp.index, imp.values, color=["#2e86c1" if v > 0 else "#999999" for v in imp.values])
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("Increase in loss when this input is shuffled (higher = model relies on it more)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Permutation importance computed locally at each hospital and averaged by the server. "
               "Small test sets make this noisy; differences between low-ranked inputs are not reliable. "
               "A value at or below zero (grey) means shuffling that input did not hurt the model, so it contributes nothing reliable.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**How it works.** A central server and three hospitals exchange only model weights. Each round the server sends the current
global model, each hospital trains it for a few epochs on its own patients, sends back the *change in weights*, and the server
averages those changes weighted by hospital size (**FedAvg**). Feature scaling statistics were also built from per-hospital
summary numbers, never from patient rows.

**Limitations (please state these in your report and viva).**
- Hospitals are *simulated* from three public UCI datasets (≈ 776 usable patients); this is not a real hospital deployment.
- The VA Long Beach data is 97 % male with 75 % disease prevalence, so the population is not representative of the general public.
- About a quarter of VA values are missing and were imputed; `ca` and `thal` were dropped because they are missing for 80–99 % of two hospitals.
- Weight updates can still leak information. Optional clipping + noise is available in the code, but no formal (ε, δ) privacy guarantee is computed.
- The public datasets contain no women under 30 and only 6 under 35 (ages 28–77 overall), so predictions for young patients are extrapolation; the app warns when an input is far outside the training range.
- "No chest pain (asymptomatic)" is associated with disease in these datasets (74 % of such patients), because people without symptoms were mostly tested when something else was suspected. The model learned this dataset pattern; it is not a clinical rule.
- Test sets are small (about 150 patients), so scores vary by a few percentage points from run to run.
""")