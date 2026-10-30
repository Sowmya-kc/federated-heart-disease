"""
run_federation.py -- starts the server and the three hospital clients as SEPARATE processes
on this computer and lets them train together over HTTP.
    python run_federation.py                      # normal run
    python run_federation.py --drop_prob 0.3 --round_timeout 4   # hospitals randomly go offline
    python run_federation.py --dp_clip 1 --dp_sigma 0.1          # noisy (privacy-protected) updates
"""
import argparse, os, subprocess, sys, time

SITES = ["Cleveland", "Hungarian", "VA Long Beach"]
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--rounds", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dp_clip", type=float, default=0.0)
ap.add_argument("--dp_sigma", type=float, default=0.0)
ap.add_argument("--drop_prob", type=float, default=0.0)
ap.add_argument("--round_timeout", type=float, default=60)
ap.add_argument("--prox_mu", type=float, default=0.0)
ap.add_argument("--out_dir", default="models")
a = ap.parse_args()
os.makedirs("logs", exist_ok=True)
py = sys.executable
server = subprocess.Popen([py, "server.py", "--port", str(a.port), "--rounds", str(a.rounds), "--seed", str(a.seed),
                           "--round_timeout", str(a.round_timeout), "--prox_mu", str(a.prox_mu),
                           "--out_dir", a.out_dir])
time.sleep(1.5)
clients = []
for s in SITES:
    f = open(f"logs/client_{s.replace(' ', '_')}.log", "w")
    clients.append(subprocess.Popen(
        [py, "client.py", "--site", s, "--server", f"http://127.0.0.1:{a.port}", "--seed", str(a.seed),
         "--dp_clip", str(a.dp_clip), "--dp_sigma", str(a.dp_sigma), "--drop_prob", str(a.drop_prob)],
        stdout=f, stderr=subprocess.STDOUT))
server.wait()
for c in clients:
    try:
        c.wait(timeout=20)
    except subprocess.TimeoutExpired:
        c.kill()
print("Hospital logs are in the logs/ folder. Trained model: " + os.path.join(a.out_dir, "global_model.npz"))
