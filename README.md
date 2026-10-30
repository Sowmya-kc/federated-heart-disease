# Privacy-Preserving Federated Learning for Heart Disease Prediction

Three hospitals (Cleveland, Hungarian, VA Long Beach; UCI Heart Disease) train one shared neural
network with Federated Averaging (FedAvg). Patient records never leave a hospital: only weights,
summary statistics and aggregate scores are exchanged.

## Quick start
    pip install -r requirements.txt
    # put cleveland.data, hungarian.data, long-beach-va.data inside ./data/
    python run_federation.py     # trains the federated model (server + 3 hospital processes), ~20 s
    streamlit run app.py         # opens the prediction web app in your browser
    python run_experiments.py    # (optional) full comparison study, 10 random splits, ~90 s

## Running the system on several computers
    # computer A (server):   python server.py
    # each hospital machine: python client.py --site "Cleveland" --server http://<IP-of-A>:8765
    #                        (sites: Cleveland | Hungarian | "VA Long Beach"; each needs only its own data file)

## Files
| file | purpose |
|---|---|
| server.py | federated server: global scaling, FedAvg, round timeouts / dropout handling, saves the model |
| client.py | one hospital: local training, optional update clipping + noise, local evaluation + feature importance |
| run_federation.py | launches the server and 3 clients locally (`--drop_prob 0.3 --round_timeout 4` simulates outages) |
| predict.py | loads `models/global_model.npz`, predicts risk + explains a single patient |
| app.py | Streamlit web app (prediction, training results, limitations) |
| model.py, federated.py, data_utils.py | NumPy neural network, simulation engine, data parsing/cleaning |
| run_experiments.py | comparison study: local vs centralized vs FedAvg vs FedProx, communication, noise, dropout |
| models/ | trained global model + training log; results/ = tables and figures |

## Data note
The supplied cleveland.data has a damaged tail: 282 of ~303 records are readable and used;
damaged records are skipped automatically.

## Disclaimer
Educational prototype. Not a medical device.
