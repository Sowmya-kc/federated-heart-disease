"""
data_utils.py  --  Loading, cleaning and splitting the three UCI Heart Disease sites.

The raw UCI files store every patient as 76 attributes spread over several text lines,
terminated by the word 'name'.  We parse them token-by-token, keep the 14 standard
attributes used in the literature, and turn the 0-4 'num' label into binary
(0 = no disease, 1 = disease present).
"""
import numpy as np
import pandas as pd

# 1-indexed attribute numbers in the raw 76-attribute UCI format
ATTR = {"age": 3, "sex": 4, "cp": 9, "trestbps": 10, "chol": 12, "fbs": 16,
        "restecg": 19, "thalach": 32, "exang": 38, "oldpeak": 40,
        "slope": 41, "ca": 44, "thal": 51, "num": 58}

SITES = {"Cleveland": "data/cleveland.data",
         "Hungarian": "data/hungarian.data",
         "VA Long Beach": "data/long-beach-va.data"}

# Features used by the model.  ca, thal and slope are dropped because they are
# missing for 50-99% of Hungarian / VA patients (see missingness_report()).
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2]}


def parse_raw(path):
    """Parse a raw UCI .data file -> DataFrame of the 14 standard attributes.
    Corrupted records (the Cleveland file has a damaged tail) are discarded."""
    toks = open(path, encoding="latin-1").read().split()
    records, cur = [], []
    for t in toks:
        if t == "name":
            records.append(cur)
            cur = []
        else:
            cur.append(t)
    good = []
    for r in records:
        if len(r) != 75:
            continue
        try:
            good.append([float(x) for x in r])
        except ValueError:
            continue
    arr = np.array(good)
    df = pd.DataFrame({k: arr[:, v - 1] for k, v in ATTR.items()})
    df = df.replace(-9, np.nan)                       # -9 = missing in UCI files
    df.loc[df["chol"] == 0, "chol"] = np.nan          # 0 cholesterol is impossible
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan  # 0 blood pressure impossible
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num"), len(records), len(good)


def load_all():
    out, info = {}, {}
    for name, p in SITES.items():
        df, n_found, n_ok = parse_raw(p)
        out[name] = df
        info[name] = (n_found, n_ok)
    return out, info


def missingness_report(sites):
    cols = NUMERIC + BINARY + list(CATEGORICAL) + ["ca", "thal"]
    rep = pd.DataFrame({n: d[cols].isna().mean() * 100 for n, d in sites.items()}).round(1)
    rep.index.name = "feature (% missing)"
    return rep


def encode(df, medians):
    """Impute with *local-or-federated* medians and one-hot encode categoricals."""
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = df[c].fillna(medians[c])
    for c in BINARY:
        X[c] = df[c].fillna(medians[c]).round()
    for c, levels in CATEGORICAL.items():
        v = df[c].fillna(medians[c])
        for lv in levels:
            X[f"{c}_{lv}"] = (v == lv).astype(float)
    return X


def split_client(df, seed, test_size=0.2):
    """Stratified train/test split done *inside* each client."""
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=test_size, stratify=df["target"],
                              random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def local_medians(train_df):
    cols = NUMERIC + BINARY + list(CATEGORICAL)
    return {c: float(train_df[c].median()) for c in cols}


def prepare_clients(sites, seed=0, federated_scaling=True):
    """
    Returns {site: dict(Xtr, ytr, Xte, yte)} as float arrays.

    Privacy-aware preprocessing:
      * missing values are imputed with the client's OWN training medians
      * standardisation uses GLOBAL mean/std built only from per-client
        sufficient statistics (n, sum, sum of squares) -- no raw records leave a client.
    """
    splits, vals = {}, {}
    for n, d in sites.items():
        tr, te = split_client(d, seed)
        tr, va = split_client(tr, seed + 1000, test_size=0.15)   # validation set for early stopping
        splits[n], vals[n] = (tr, te), va
    enc_tr, enc_te, enc_va = {}, {}, {}
    for n, (tr, te) in splits.items():
        med = local_medians(tr)
        enc_tr[n], enc_te[n], enc_va[n] = encode(tr, med), encode(te, med), encode(vals[n], med)
    cols = list(next(iter(enc_tr.values())).columns)

    if federated_scaling:   # aggregate sufficient statistics only
        N = sum(len(x) for x in enc_tr.values())
        s = sum(x[NUMERIC].sum() for x in enc_tr.values())
        ss = sum((x[NUMERIC] ** 2).sum() for x in enc_tr.values())
        mu = s / N
        sd = np.sqrt(np.maximum(ss / N - mu ** 2, 1e-12))
    out = {}
    for n in splits:
        a, b, v = enc_tr[n].copy(), enc_te[n].copy(), enc_va[n].copy()
        if not federated_scaling:
            mu, sd = a[NUMERIC].mean(), a[NUMERIC].std() + 1e-8
        for z in (a, b, v):
            z[NUMERIC] = (z[NUMERIC] - mu) / sd
        out[n] = dict(Xtr=a.values.astype(float), ytr=splits[n][0]["target"].values.astype(float),
                      Xva=v.values.astype(float), yva=vals[n]["target"].values.astype(float),
                      Xte=b.values.astype(float), yte=splits[n][1]["target"].values.astype(float))
    return out, cols
