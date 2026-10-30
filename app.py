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

st.title("🫀 Privacy-Preserving Federated Heart Disease Prediction")
st.caption("CMR University capstone · model trained by 3 simulated hospitals (Cleveland, Hungarian, VA Long Beach) "
           "using Federated Averaging. No patient record was ever shared.")

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
        if risk >= 0.6:
            st.warning("The model estimates a **high** likelihood of heart disease for this input.")
        elif risk >= 0.3:
            st.info("The model estimates a **moderate** likelihood for this input.")
        else:
            st.success("The model estimates a **low** likelihood for this input.")
        st.caption("Bands (low / moderate / high) are illustrative only, not clinical thresholds.")

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
    st.dataframe(pc.drop(columns="loss").round(3), width="stretch")

    st.subheader("Training progress (federated validation loss per round)")
    vh = pd.Series(log["val_loss_history"], name="validation loss")
    vh.index.name = "communication round"
    st.line_chart(vh)
    st.caption(f"The server kept the global model from round {log['best_round']} (lowest validation loss). "
               "Validation loss is computed by the hospitals on their own data.")

    st.subheader("Which inputs does the federated model rely on?")
    imp = pd.Series(log["importance"]).sort_values()
    st.bar_chart(imp.rename("increase in loss when the input is shuffled"))
    st.caption("Permutation importance computed locally at each hospital and averaged by the server. "
               "Small test sets make this noisy; differences between low-ranked inputs are not reliable.")

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
- Test sets are small (about 150 patients), so scores vary by a few percentage points from run to run.
""")
