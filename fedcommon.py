"""fedcommon.py -- helpers shared by the federated server, clients and predictor."""
import json
import urllib.request
from data_utils import NUMERIC, BINARY, CATEGORICAL

# original clinical feature -> the model-input columns it produces after encoding
FEATURE_GROUPS = {c: [c] for c in NUMERIC + BINARY}
FEATURE_GROUPS.update({c: [f"{c}_{lv}" for lv in levels] for c, levels in CATEGORICAL.items()})
N_FEATURES = sum(len(v) for v in FEATURE_GROUPS.values())      # 15


def http_get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def http_post(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
