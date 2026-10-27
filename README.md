# Privacy-Preserving Federated Learning for Heart Disease Prediction

Three simulated hospitals (Cleveland, Hungarian, VA Long Beach; UCI Heart Disease) train a
neural network collaboratively with Federated Averaging (FedAvg). Raw patient records never
leave a client; only model weights are exchanged.

## Run
    pip install -r requirements.txt
    # put cleveland.data, hungarian.data, long-beach-va.data inside ./data/
    python run_experiments.py        # ~90 s, writes all tables + figures to ./results/

## Files
- data_utils.py        parsing of the raw UCI files, cleaning, per-client split, privacy-aware preprocessing
- model.py             MLP (15 -> 16 -> 8 -> 1) in pure NumPy, with FedProx proximal term
- federated.py         FedAvg engine, centralized baseline, local-only baseline, DP-style noise, client dropout
- run_experiments.py   every experiment in the report (10 random splits each)
- results/             CSV tables (01-11) and figures (fig1-fig7)

## Data note
The supplied cleveland.data has a damaged tail: 282 of ~303 records are readable and
are used; damaged records are skipped automatically.
