"""
IoT-23 loader — a second, independent real dataset (Stratosphere Lab, CTU).

Source: https://www.stratosphereips.org/datasets-iot23 — only the labelled Zeek
flow logs (bro/conn.log.labeled) are used, never the PCAPs. Captures in use,
under data/iot23/<capture>/conn.log.labeled:

    CTU-IoT-Malware-Capture-34-1   Mirai: 14,394 DDoS + 6,706 C&C + 1,923 benign flows
    CTU-Honeypot-Capture-5-1       Amazon Echo, 1,374 benign flows
    CTU-Honeypot-Capture-4-1       Philips HUE, 452 benign flows (parsed, not used:
                                   too small for either scenario's benign budget)

Unlike N-BaIoT (115 pre-aggregated statistics per packet window), each IoT-23
record is ONE flow, so the features are per-flow: byte/packet counts, duration,
and one-hot protocol / connection state. Same (features, label, meta) tuples as
the other loaders, same phases, so run_scenarios.py is agnostic to the source.

Format quirks handled here (measured on the downloaded files):
  - honeypot logs separate the last three columns with spaces, not tabs
  - malware logs call the last column `det_label`, honeypots `detailed-label`
  - label case differs ("benign" vs "Benign")
  - "-" marks an unset numeric field -> 0

meta["reputation"] is, as for N-BaIoT, keyed to ground truth (benign 1.0,
attack 0.05) — a simulated channel. IoT-23 does carry destination IPs
(id.resp_h), so a REAL reputation lookup is possible here in principle;
that is Project-II work.

Scenario pacing is constructed exactly as in load_nbaiot.py: real attack flows
are walked toward real benign flows in signed-log space.
"""
import os

import numpy as np
import pandas as pd

from data.load_nbaiot import _interp

DEFAULT_ROOT = os.environ.get(
    "IOT23_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "iot23"))
MIRAI = "CTU-IoT-Malware-Capture-34-1"
BENIGN_B = "CTU-Honeypot-Capture-5-1"

BENIGN_REPUTATION = 1.0
ATTACK_REPUTATION = 0.05

NUMERIC = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
           "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
PROTOS = ["tcp", "udp", "icmp"]
CONN_STATES = ["S0", "S1", "SF", "REJ", "RSTO", "RSTR", "RSTOS0", "SH", "SHR", "OTH"]

_CACHE = {}


def feature_names():
    return (NUMERIC + [f"proto_{p}" for p in PROTOS]
            + [f"state_{s}" for s in CONN_STATES])


def parse_conn_log(path):
    """conn.log.labeled -> DataFrame with feature_names() + `label` (0/1) + `detail`."""
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"IoT-23 log not found: {path}\n"
            "Download bro/conn.log.labeled for the capture from "
            "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/")
    fields, rows = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                fields = line.split()[1:]          # split() copes with the space quirk
            elif line and not line.startswith("#"):
                rows.append(line.split())
    if fields is None:
        raise ValueError(f"{path}: no #fields header — not a Zeek conn log")
    out = featurize(fields, rows)
    _CACHE[path] = out
    return out


def featurize(fields, rows):
    """Zeek conn records (header field names + split rows) -> feature_names() columns,
    plus `label` / `detail` when the log is labelled (IoT-23). Used for files and for
    live lines sent to the backend (POST /api/live/ingest); unlabelled live lines get
    label -1."""
    fields = ["detailed_label" if c in ("det_label", "detailed-label") else c for c in fields]
    rows = [r for r in rows if len(r) == len(fields)]
    raw = pd.DataFrame(rows, columns=fields)

    out = pd.DataFrame(index=raw.index)
    for c in NUMERIC:
        out[c] = pd.to_numeric(raw[c].replace("-", "0"), errors="coerce").fillna(0.0)
    for p in PROTOS:
        out[f"proto_{p}"] = (raw["proto"] == p).astype(float)
    for s in CONN_STATES:
        out[f"state_{s}"] = (raw["conn_state"] == s).astype(float)
    if "label" in raw:
        out["label"] = (raw["label"].str.lower() == "malicious").astype(int)
        out["detail"] = raw["detailed_label"] if "detailed_label" in raw else "-"
    else:
        out["label"] = -1
        out["detail"] = "-"
    return out


def parse_timestamps(path):
    """Flow start times (Zeek `ts`, epoch seconds), row-aligned with parse_conn_log(path)."""
    fields, ts = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                fields = line.split()[1:]
            elif line and not line.startswith("#"):
                r = line.split()
                if fields is not None and len(r) == len(fields):
                    ts.append(float(r[fields.index("ts")]))
    return np.array(ts)


def _log(capture, root):
    return parse_conn_log(os.path.join(root or DEFAULT_ROOT, capture, "conn.log.labeled"))


def _take(df, n, offset, what):
    if offset + n > len(df):
        raise ValueError(f"{what}: has {len(df)} flows, requested {n} from offset {offset}")
    return df.iloc[offset:offset + n][feature_names()].to_numpy(dtype=float)


def _record(vec, label, reputation, src):
    return ({k: float(v) for k, v in zip(feature_names(), vec)}, label,
            {"reputation": reputation, "source": src})


def _pools(root):
    """Benign flows of the Mirai capture (device A) and its attack flows, in time order."""
    m = _log(MIRAI, root)
    return m[m.label == 0], m[m.label == 1]


def calibration_block(n_calib, attack_fraction, benign, attack, seed):
    n_attack = int(n_calib * attack_fraction)
    block = ([_record(v, 0, BENIGN_REPUTATION, "calib_benign")
              for v in _take(benign, n_calib - n_attack, 0, "benign calib")]
             + [_record(v, 1, ATTACK_REPUTATION, "calib_attack")
                for v in _take(attack, n_attack, 0, "attack calib")])
    idx = np.random.default_rng(seed).permutation(len(block))
    return [block[i] for i in idx]


def slow_poisoning_stream(n_calib=400, n_pre=400, n_poison=600, n_post=400,
                          n_recovery_check=200, seed=42, root=None):
    benign, attack = _pools(root)
    stream = calibration_block(n_calib, 0.3, benign, attack, seed)
    b_off, a_off = n_calib - int(n_calib * 0.3), int(n_calib * 0.3)

    stream += [_record(v, 0, BENIGN_REPUTATION, "pre_attack")
               for v in _take(benign, n_pre, b_off, "benign pre")]
    # benign is the scarce pool (1,923 flows): calib 280 + pre 400 + targets
    # 1,000 = 1,680, so every slice is still disjoint, as in load_nbaiot.py
    targets = _take(benign, n_poison + n_post, b_off + n_pre, "benign targets")
    atk = _take(attack, n_poison, a_off, "attack poison")
    for i in range(n_poison):
        alpha = i / max(1, n_poison - 1)
        stream.append(_record(_interp(atk[i], targets[i], alpha), 1, ATTACK_REPUTATION,
                              "slow_poison"))
    atk_p = _take(attack, n_post, a_off + n_poison, "attack post")
    for i in range(n_post):
        stream.append(_record(_interp(atk_p[i], targets[n_poison + i], 1.0), 1,
                              ATTACK_REPUTATION, "post_poison"))
    stream += [_record(v, 1, ATTACK_REPUTATION, "recovery_check")
               for v in _take(attack, n_recovery_check, a_off + n_poison + n_post,
                              "attack recovery")]
    return stream


def genuine_drift_stream(n_calib=400, n_pre=400, n_drift=600, n_post=400, seed=42,
                         root=None):
    """Device A = benign flows of the Mirai-capture network, device B = Amazon Echo."""
    benign, attack = _pools(root)
    b = _log(BENIGN_B, root)
    b = b[b.label == 0]
    stream = calibration_block(n_calib, 0.3, benign, attack, seed)
    off = n_calib - int(n_calib * 0.3)
    stream += [_record(v, 0, BENIGN_REPUTATION, "pre_drift")
               for v in _take(benign, n_pre, off, "benign A pre")]
    a_rows = _take(benign, n_drift, off + n_pre, "benign A drift")
    b_rows = _take(b, n_drift, 0, "benign B drift")
    for i in range(n_drift):
        alpha = i / max(1, n_drift - 1)
        stream.append(_record(_interp(a_rows[i], b_rows[i], alpha), 0, BENIGN_REPUTATION,
                              "genuine_drift"))
    stream += [_record(v, 0, BENIGN_REPUTATION, "post_drift")
               for v in _take(b, n_post, n_drift, "benign B post")]
    return stream


def available(root=None):
    return all(os.path.exists(os.path.join(root or DEFAULT_ROOT, c, "conn.log.labeled"))
               for c in (MIRAI, BENIGN_B))
