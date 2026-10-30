"""
predict.py -- load the federated global model and score a new patient.
    from predict import HeartRiskModel
    m = HeartRiskModel()
    m.predict_proba(dict(age=58, sex=1, cp=4, trestbps=140, chol=240, fbs=0, restecg=0,
                         thalach=130, exang=1, oldpeak=2.0))
"""
import json
import numpy as np
import pandas as pd
from data_utils import NUMERIC, BINARY, CATEGORICAL, encode
from model import MLP

FIELDS = NUMERIC + BINARY + list(CATEGORICAL)       # the 10 clinical inputs


class HeartRiskModel:
    def __init__(self, path="models/global_model.npz"):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.net = MLP(self.meta["n_in"], tuple(self.meta["hidden"]))
        self.net.set_params(z["weights"])
        self.mu = np.array(self.meta["scaler"]["mu"])
        self.sd = np.array(self.meta["scaler"]["sd"])
        self.medians = self.meta["medians"]

    def _features(self, patients):
        df = pd.DataFrame(patients)
        for c in FIELDS:                              # absent / None -> NaN -> imputed with global median
            if c not in df:
                df[c] = np.nan
        X = encode(df, self.medians)
        X[NUMERIC] = (X[NUMERIC] - self.mu) / self.sd
        return X.values.astype(float)

    def predict_proba(self, patient):
        """Probability (0-1) that the patient has heart disease."""
        return float(self.net.predict_proba(self._features([patient]))[0])

    def explain(self, patient):
        """For each input: how many probability points it adds/removes compared with a 'typical'
        (median) value of that input, all other inputs unchanged. Positive = raises the risk."""
        p0 = self.predict_proba(patient)
        out = {}
        for f in FIELDS:
            alt = dict(patient); alt[f] = self.medians[f]
            out[f] = p0 - self.predict_proba(alt)
        return out
