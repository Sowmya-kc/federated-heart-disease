"""
client.py -- one hospital.  Its patient file is read LOCALLY and is never sent anywhere.
What leaves this program: summary statistics, weight updates, validation-loss numbers,
and final test metrics / feature importances (all aggregate numbers).
    python client.py --site Cleveland --server http://127.0.0.1:8765
"""
import argparse, time, urllib.error, urllib.parse
import numpy as np
from data_utils import SITES, NUMERIC, parse_raw, split_client, local_medians, encode
from model import MLP
from federated import metrics
from fedcommon import http_get, http_post, FEATURE_GROUPS, N_FEATURES


def importance(model, X, y, cols, rng, reps=5):
    """Permutation importance: how much does the test loss rise when one clinical feature is shuffled?"""
    base, out = model.loss(X, y), {}
    for g, names in FEATURE_GROUPS.items():
        idx = [cols.index(c) for c in names]
        d = []
        for _ in range(reps):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            d.append(model.loss(Xp, y) - base)
        out[g] = float(np.mean(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dp_clip", type=float, default=0.0, help="clip update norm (0 = off)")
    ap.add_argument("--dp_sigma", type=float, default=0.0, help="Gaussian noise multiplier")
    ap.add_argument("--drop_prob", type=float, default=0.0, help="simulate going offline in a round")
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 8])
    a = ap.parse_args()
    cid = a.site.replace(" ", "_")
    say = lambda m: print(f"[{a.site}] {m}", flush=True)

    # ---------------- local data (never leaves this process) ----------------
    df, _, n_ok = parse_raw(SITES[a.site])
    tr, te = split_client(df, a.seed)
    tr, va = split_client(tr, a.seed + 1000, test_size=0.15)
    med = local_medians(tr)
    E = {"tr": encode(tr, med), "va": encode(va, med), "te": encode(te, med)}
    y = {"tr": tr["target"].values.astype(float), "va": va["target"].values.astype(float),
         "te": te["target"].values.astype(float)}
    cols = list(E["tr"].columns)
    say(f"loaded {n_ok} local patients ({len(tr)} train / {len(va)} val / {len(te)} test)")
    rng = np.random.default_rng(a.seed + sum(map(ord, cid)))
    model = MLP(N_FEATURES, tuple(a.hidden), seed=a.seed)

    def scaled(split, scaler):
        X = E[split].copy()
        X[NUMERIC] = (X[NUMERIC] - np.array(scaler["mu"])) / np.array(scaler["sd"])
        return X.values.astype(float)

    last, fails = "", 0
    while True:
        try:
            task = http_get(f"{a.server}/task?client={urllib.parse.quote(cid)}&last={last}")
            fails = 0
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            fails += 1
            if fails > 40:
                say("server not reachable - giving up"); return
            time.sleep(0.5); continue
        if task.get("wait"):
            time.sleep(0.2); continue
        ph, tok = task["phase"], task["token"]
        if ph == "done":
            say("federation finished"); return
        post = lambda payload: http_post(f"{a.server}/submit", dict(client=cid, token=tok, payload=payload))

        if ph == "stats":
            s = E["tr"][NUMERIC]
            post(dict(n=len(s), sum=s.sum().tolist(), sumsq=(s ** 2).sum().tolist(), medians=med))
            say("sent summary statistics (no patient rows)")
        else:
            sc, h = task["scaler"], task["hyper"]
            w = np.array(task["weights"]); model.set_params(w)
            if ph == "train":
                Xtr, Xva = scaled("tr", sc), scaled("va", sc)
                vl = model.loss(Xva, y["va"])                       # score of the model we RECEIVED
                if rng.random() < a.drop_prob:
                    say(f"round {task['round']}: offline (simulated)"); last = tok; continue
                model.sgd_epochs(Xtr, y["tr"], h["epochs"], h["lr"], h["batch"], h["l2"], rng=rng,
                                 mu=h["prox_mu"], w_global=w)
                delta = model.get_params() - w
                if a.dp_clip:
                    delta *= min(1.0, a.dp_clip / (np.linalg.norm(delta) + 1e-12))
                    delta += rng.normal(0, a.dp_sigma * a.dp_clip, delta.shape)
                post(dict(delta=delta.tolist(), n_train=len(Xtr), val_loss=float(vl), n_val=len(Xva)))
                say(f"round {task['round']}: trained locally, sent weight update")
            elif ph == "eval":
                Xva = scaled("va", sc)
                post(dict(val_loss=float(model.loss(Xva, y["va"])), n_val=len(Xva)))
            elif ph == "test":
                Xte = scaled("te", sc)
                m = metrics(model, Xte, y["te"])
                pred = (model.predict_proba(Xte) >= 0.5).astype(int)
                yt = y["te"].astype(int)
                post(dict(n=len(Xte), acc=m["acc"], prec=m["prec"], rec=m["rec"], f1=m["f1"], auc=m["auc"],
                          loss=m["loss"], tp=int(((pred == 1) & (yt == 1)).sum()), fp=int(((pred == 1) & (yt == 0)).sum()),
                          tn=int(((pred == 0) & (yt == 0)).sum()), fn=int(((pred == 0) & (yt == 1)).sum()),
                          importance=importance(model, Xte, y["te"], cols, rng)))
                say(f"final local test accuracy of the global model: {m['acc']:.3f}")
        last = tok


if __name__ == "__main__":
    main()
