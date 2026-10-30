"""
server.py -- the central federated-learning server (standard library only).

The server NEVER sees patient data.  It only receives:
   * summary statistics (n, sum, sum of squares) used to standardise features globally
   * model-weight updates  (and validation-loss numbers) from every hospital
Protocol (phases):  stats -> train (round 1..R) -> eval -> test -> done
   train : broadcast global weights, collect client updates, aggregate with FedAvg
   eval  : clients only score the final global weights on their validation data
   test  : the best round (lowest federated validation loss) is sent to the clients, who report
           test metrics + feature importance computed LOCALLY; the server pools those numbers.
Fault tolerance: a round closes when all clients answered, or when --round_timeout seconds have
passed and at least --min_clients answered.
"""
import argparse, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from model import MLP
from data_utils import NUMERIC, BINARY, CATEGORICAL
from fedcommon import N_FEATURES, FEATURE_GROUPS


class Coordinator:
    def __init__(self, a):
        self.a, self.lock = a, threading.Lock()
        self.phase, self.round, self.attempt, self.token = "stats", 0, 0, "stats:0:0"
        self.received, self.started = {}, time.time()
        self.scaler = self.medians = None
        self.w_hist, self.val_hist, self.best = [], [], None
        self.log = dict(config=vars(a), rounds=[], per_client_test={}, pooled_test={}, importance={})
        self.bytes_in = self.bytes_out = 0
        self.done_time = None
        self.final_w = None

    # ------------------------------------------------------------------ tasks
    def task_for(self, client, last):
        with self.lock:
            if self.token == last:
                return {"wait": True}
            t = {"token": self.token, "phase": self.phase, "round": self.round, "scaler": self.scaler,
                 "hyper": dict(epochs=self.a.epochs, lr=self.a.lr, batch=self.a.batch, l2=self.a.l2,
                               prox_mu=self.a.prox_mu)}
            if self.phase in ("train", "eval"):
                t["weights"] = self.w_hist[-1].tolist()
            elif self.phase == "test":
                t["weights"] = self.final_w.tolist()
            return t

    def submit(self, client, token, payload):
        with self.lock:
            if token != self.token:
                return {"ok": False, "stale": True}
            self.received[client] = payload
            self._maybe_advance()
            return {"ok": True}

    def tick(self):
        with self.lock:
            self._maybe_advance()

    def _maybe_advance(self):
        n = len(self.received)
        late = time.time() - self.started > self.a.round_timeout
        if self.phase == "done":
            return
        if n >= self.a.n_clients or (n >= self.a.min_clients and late):
            self._advance()
        elif late and self.phase != "stats":
            # too few hospitals answered in time: re-issue the same round under a new token
            self.attempt += 1
            self.token = f"{self.phase}:{self.round}:{self.attempt}"
            self.received, self.started = {}, time.time()
            print(f"[server] round {self.round}: only {n} hospital(s) answered - re-issuing the round", flush=True)

    # ------------------------------------------------------------- state machine
    def _go(self, phase, rnd):
        self.phase, self.round, self.attempt = phase, rnd, 0
        self.token = f"{phase}:{rnd}:0"
        self.received, self.started = {}, time.time()

    def _advance(self):
        rc, a = self.received, self.a
        if self.phase == "stats":
            N = sum(p["n"] for p in rc.values())
            s = sum(np.array(p["sum"]) for p in rc.values())
            ss = sum(np.array(p["sumsq"]) for p in rc.values())
            mu = s / N
            sd = np.sqrt(np.maximum(ss / N - mu ** 2, 1e-12))
            self.scaler = {"mu": mu.tolist(), "sd": sd.tolist()}
            med = {}
            for c in NUMERIC:                       # weighted mean of the clients' medians
                med[c] = sum(p["n"] * p["medians"][c] for p in rc.values()) / N
            for c in BINARY + list(CATEGORICAL):     # weighted vote for discrete features
                votes = {}
                for p in rc.values():
                    votes[p["medians"][c]] = votes.get(p["medians"][c], 0) + p["n"]
                med[c] = max(votes, key=votes.get)
            self.medians = med
            self.w_hist = [MLP(N_FEATURES, tuple(a.hidden), seed=a.seed).get_params()]
            print(f"[server] global scaling built from {len(rc)} clients ({N} training patients)", flush=True)
            self._go("train", 1)

        elif self.phase == "train":
            r = self.round
            sizes = np.array([p["n_train"] for p in rc.values()], float)
            vl = sum(p["val_loss"] * p["n_val"] for p in rc.values()) / sum(p["n_val"] for p in rc.values())
            self.val_hist.append(vl)                           # loss of w_{r-1} (what clients received)
            w = self.w_hist[-1] + sum(s / sizes.sum() * np.array(p["delta"])
                                      for s, p in zip(sizes, rc.values()))        # <-- FedAvg
            self.w_hist.append(w)
            self.log["rounds"].append(dict(round=r, clients=sorted(rc), val_loss_before_round=vl,
                                           MB_so_far=(self.bytes_in + self.bytes_out) / 1e6))
            print(f"[server] round {r:2d}/{a.rounds}: {len(rc)}/{a.n_clients} clients | "
                  f"val loss of received model {vl:.4f}", flush=True)
            self._go("train", r + 1) if r < a.rounds else self._go("eval", a.rounds + 1)

        elif self.phase == "eval":
            vl = sum(p["val_loss"] * p["n_val"] for p in rc.values()) / sum(p["n_val"] for p in rc.values())
            self.val_hist.append(vl)
            self.best = int(np.argmin(self.val_hist))           # index i = weights after i rounds
            self.final_w = self.w_hist[self.best]
            self.log["best_round"] = self.best
            self.log["val_loss_history"] = [float(v) for v in self.val_hist]
            print(f"[server] best global model = after round {self.best} (val loss {self.val_hist[self.best]:.4f})",
                  flush=True)
            self._go("test", self.best)

        elif self.phase == "test":
            tp, fp, tn, fn = (sum(p[k] for p in rc.values()) for k in ("tp", "fp", "tn", "fn"))
            n = tp + fp + tn + fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            self.log["pooled_test"] = dict(
                n=n, accuracy=(tp + tn) / n, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                auc_mean_of_clients=sum(p["auc"] * p["n"] for p in rc.values()) / n,
                confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn))
            self.log["per_client_test"] = {c: {k: p[k] for k in ("n", "acc", "prec", "rec", "f1", "auc", "loss")}
                                           for c, p in rc.items()}
            self.log["importance"] = {g: sum(p["n"] * p["importance"][g] for p in rc.values()) / n
                                      for g in FEATURE_GROUPS}
            self._finish()

    def _finish(self):
        a = self.a
        os.makedirs(a.out_dir, exist_ok=True)
        meta = dict(hidden=list(a.hidden), n_in=N_FEATURES, scaler=self.scaler, medians=self.medians,
                    best_round=self.best, rounds=a.rounds)
        np.savez(os.path.join(a.out_dir, "global_model.npz"), weights=self.final_w, meta=np.array(json.dumps(meta)))
        self.log["traffic_MB_total"] = (self.bytes_in + self.bytes_out) / 1e6
        json.dump(self.log, open(os.path.join(a.out_dir, "training_log.json"), "w"), indent=2, default=float)
        p = self.log["pooled_test"]
        print(f"[server] FINAL (federated test, {p['n']} patients): acc {p['accuracy']:.3f}  prec {p['precision']:.3f}  "
              f"rec {p['recall']:.3f}  f1 {p['f1']:.3f}", flush=True)
        print(f"[server] total network traffic {self.log['traffic_MB_total']:.2f} MB. "
              f"Model saved to {a.out_dir}/global_model.npz", flush=True)
        self.phase, self.token, self.done_time = "done", "done", time.time()


def make_handler(co):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            co.bytes_out += len(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/task":
                if co.phase == "done":
                    return self._send({"phase": "done", "token": "done"})
                return self._send(co.task_for(q["client"][0], q.get("last", [""])[0]))
            self._send({"phase": co.phase, "round": co.round})

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            co.bytes_in += n
            d = json.loads(self.rfile.read(n))
            self._send(co.submit(d["client"], d["token"], d["payload"]))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--n_clients", type=int, default=3)
    ap.add_argument("--min_clients", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--round_timeout", type=float, default=60)
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 8])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--prox_mu", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="models")
    a = ap.parse_args()
    co = Coordinator(a)
    httpd = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(co))

    def monitor():
        while True:
            time.sleep(0.5)
            co.tick()
            if co.done_time and time.time() - co.done_time > 3:
                httpd.shutdown()
                return
    threading.Thread(target=monitor, daemon=True).start()
    print(f"[server] listening on port {a.port}; waiting for {a.n_clients} hospitals ...", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
