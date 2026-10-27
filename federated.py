"""
federated.py -- Federated Averaging (FedAvg) engine + baselines.

One communication round:
  1. server broadcasts global weights w_t
  2. every available client trains locally for E epochs on its OWN data
  3. clients send back only their weights (never records)
  4. server aggregates:  w_{t+1} = sum_k (n_k / n) * w_k          <-- FedAvg
Optional extras (all map to the 'Identified Gaps' slide):
  * mu > 0         : FedProx proximal term (non-IID data)
  * dp_clip/dp_sigma: clip each update and add Gaussian noise (update privacy)
  * dropout        : probability that a client is offline in a round (fault tolerance)
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import MLP

HP = dict(hidden=(16, 8), lr=0.03, batch=32, l2=1e-3, local_epochs=3, rounds=40)


def metrics(model, X, y):
    p = model.predict_proba(X)
    yhat = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat),
                prec=precision_score(y, yhat, zero_division=0),
                rec=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                loss=model.loss(X, y))


def pool(clients, split):
    X = np.vstack([c[f"X{split}"] for c in clients.values()])
    y = np.concatenate([c[f"y{split}"] for c in clients.values()])
    return X, y


def eval_everywhere(model, clients):
    """metrics on every client's test set + the pooled test set."""
    res = {n: metrics(model, c["Xte"], c["yte"]) for n, c in clients.items()}
    Xp, yp = pool(clients, "te")
    res["Pooled"] = metrics(model, Xp, yp)
    return res


def run_fedavg(clients, seed=0, rounds=None, local_epochs=None, mu=0.0,
               dp_clip=None, dp_sigma=0.0, dropout=0.0, hp=HP, track=True):
    rounds = rounds or hp["rounds"]
    E = local_epochs or hp["local_epochs"]
    rng = np.random.default_rng(seed)
    n_in = next(iter(clients.values()))["Xtr"].shape[1]
    glob = MLP(n_in, hp["hidden"], seed=seed)
    w = glob.get_params()
    names = list(clients)
    history, bytes_sent = [], 0
    Xte, yte = pool(clients, "te")
    local = MLP(n_in, hp["hidden"], seed=seed)
    Xva, yva = pool(clients, "va")       # = federated evaluation: clients only report (loss, n)
    best_loss, best_w, best_r = np.inf, w.copy(), 0
    for r in range(1, rounds + 1):
        active = [n for n in names if rng.random() >= dropout]
        if not active:                       # nobody online: skip the round
            active = [names[rng.integers(len(names))]]
        updates, sizes = [], []
        for n in active:
            c = clients[n]
            local.set_params(w)
            local.sgd_epochs(c["Xtr"], c["ytr"], E, hp["lr"], hp["batch"], hp["l2"],
                             rng=rng, mu=mu, w_global=w)
            delta = local.get_params() - w
            if dp_clip:                      # clip update norm, then add Gaussian noise
                delta *= min(1.0, dp_clip / (np.linalg.norm(delta) + 1e-12))
                delta += rng.normal(0, dp_sigma * dp_clip, delta.shape)
            updates.append(delta); sizes.append(len(c["Xtr"]))
            bytes_sent += 2 * delta.size * 8           # download + upload, float64
        sizes = np.array(sizes, float)
        w = w + sum(s / sizes.sum() * d for s, d in zip(sizes, updates))   # FedAvg
        glob.set_params(w)
        vl = glob.loss(Xva, yva)
        if vl < best_loss:                   # early stopping: remember best round
            best_loss, best_w, best_r = vl, w.copy(), r
        if track:
            m = metrics(glob, Xte, yte)
            m.update(round=r, MB=bytes_sent / 1e6, active=len(active))
            history.append(m)
    glob.set_params(best_w)
    glob.best_round = best_r
    return glob, history


def run_centralized(clients, seed=0, epochs=None, hp=HP, track=True):
    """Upper-bound baseline: all training data pooled in one place (violates privacy)."""
    rounds = hp["rounds"] * hp["local_epochs"] if epochs is None else epochs
    Xtr, ytr = pool(clients, "tr")
    Xte, yte = pool(clients, "te")
    rng = np.random.default_rng(seed)
    m = MLP(Xtr.shape[1], hp["hidden"], seed=seed)
    Xva, yva = pool(clients, "va")
    best_loss, best_w, hist, best_r = np.inf, m.get_params(), [], 0
    for e in range(1, rounds + 1):
        m.sgd_epochs(Xtr, ytr, 1, hp["lr"], hp["batch"], hp["l2"], rng=rng)
        if e % hp["local_epochs"] == 0:                  # check once per 'round-equivalent'
            vl = m.loss(Xva, yva)
            if vl < best_loss: best_loss, best_w, best_r = vl, m.get_params(), e // hp["local_epochs"]
            if track:
                h = metrics(m, Xte, yte); h["round"] = e // hp["local_epochs"]; hist.append(h)
    m.set_params(best_w)
    m.best_round = best_r
    return m, hist


def run_local_only(clients, seed=0, hp=HP):
    """Each hospital trains alone for the same total number of epochs."""
    rng = np.random.default_rng(seed)
    out = {}
    for n, c in clients.items():
        m = MLP(c["Xtr"].shape[1], hp["hidden"], seed=seed)
        best_loss, best_w = np.inf, m.get_params()
        for _ in range(hp["rounds"]):                    # same total epochs, same early stopping
            m.sgd_epochs(c["Xtr"], c["ytr"], hp["local_epochs"], hp["lr"], hp["batch"], hp["l2"], rng=rng)
            vl = m.loss(c["Xva"], c["yva"])
            if vl < best_loss: best_loss, best_w = vl, m.get_params()
        m.set_params(best_w)
        out[n] = m
    return out
