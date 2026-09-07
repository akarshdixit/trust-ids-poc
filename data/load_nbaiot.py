"""
Real N-BaIoT loader — the Review-2 data path.

Dataset: https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
Device in use: Danmini_Doorbell (has both Mirai and Gafgyt families).
Each CSV ships 115 pre-extracted numeric features, so there is no pcap/Zeek
parsing step.

VERIFIED ON THE DOWNLOADED COPY (check_before_real_data.py + probes):
    benign_traffic.csv          49,548 rows x 115 cols   0 NaN   0 Inf
    mirai_attacks/scan.csv     107,685 rows x 115 cols   0 NaN   0 Inf
    gafgyt_attacks/combo.csv    59,718 rows x 115 cols   0 NaN   0 Inf
    column order identical across all three files
    20 columns (HH_*/HpHp_* covariance and pcc) take negative values
    feature dynamic range spans ~19 orders of magnitude (max 5.67e17)

This module yields the same (features_dict, label, meta_dict) tuples as
synth_stream.py, so experiments/run_scenarios.py is agnostic to the source.

meta["reputation"] is the out-of-band channel and has NO N-BaIoT equivalent
— the dataset carries no destination/reputation field at all. It is keyed
here to ground-truth traffic origin (benign -> 1.0, attack -> 0.05). Label
it in the report as a SIMULATED out-of-band signal, proof-of-concept. The
defensible claim is "we demonstrate what an authorization mechanism does
once it has an independent channel", NOT "the gate detects the attacker".
"""
import os

import numpy as np
import pandas as pd

_CACHE = {}

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbaiot", "Danmini_Doorbell")
BENIGN_CSV = os.path.join(BASE, "benign_traffic.csv")
MIRAI_SCAN_CSV = os.path.join(BASE, "mirai_attacks", "scan.csv")
GAFGYT_COMBO_CSV = os.path.join(BASE, "gafgyt_attacks", "combo.csv")

BENIGN_REPUTATION = 1.0
ATTACK_REPUTATION = 0.05


def _read(path):
    """Read once, cache — these files are 46-107 MB and get reused per scenario."""
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"N-BaIoT file not found: {path}\n"
                "Download Danmini_Doorbell from the UCI link at the top of this file."
            )
        _CACHE[path] = pd.read_csv(path)
    return _CACHE[path]


def feature_names(path=None):
    """The 115 real column names, in file order. Replaces synth FEATURE_NAMES."""
    return list(_read(path or BENIGN_CSV).columns)


FEATURE_NAMES = None  # populated lazily by feature_names(); see run_scenarios.py


def _rows(path, n, offset=0, rng=None):
    """Take n rows from a file as a float matrix, starting at `offset`."""
    df = _read(path)
    if offset + n > len(df):
        raise ValueError(
            f"{os.path.basename(path)} has {len(df)} rows, "
            f"requested {n} from offset {offset}"
        )
    return df.iloc[offset:offset + n].to_numpy(dtype=float)


def _record(vec, names, label, reputation, src):
    feats = {name: float(v) for name, v in zip(names, vec)}
    return feats, label, {"reputation": reputation, "source": src}


# --- signed-log interpolation ---------------------------------------------
# Linear interpolation in RAW feature space is useless here: the five
# HH_jit_*_variance columns (order 1e17) carry ~100% of the vector norm, so a
# raw-space path from attack to benign stays visually "attack" for almost the
# whole trajectory and then collapses at the very end. That is not a SLOW
# poisoning attack, it is a sudden one with a long flat prefix.
#
# A patient attacker paces the change as *perceived* by the detector, and the
# detector standardises. So we interpolate in signed-log space — the same space
# evidence/featurespace.py measures in — then invert back to raw values, so the
# IDS still receives realistically-scaled N-BaIoT numbers.

def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _inv_slog(z):
    return np.sign(z) * np.expm1(np.abs(z))


def _interp(a_raw, b_raw, alpha):
    """alpha=0 -> a_raw, alpha=1 -> b_raw, geodesic in signed-log space."""
    return _inv_slog((1.0 - alpha) * _slog(a_raw) + alpha * _slog(b_raw))


# --- calibration ----------------------------------------------------------

def calibration_block(n_calib=400, attack_fraction=0.3, attack_csv=None,
                      benign_offset=0, attack_offset=0, seed=42):
    """
    Labelled bootstrap set. A real IDS is never trained from zero — it starts
    from a historical labelled corpus containing known attacks alongside
    benign traffic. Without this the model has no concept of "attack" and
    every downstream scenario is vacuous.

    Mirrors synth_stream._calibration_block, with real rows.
    """
    attack_csv = attack_csv or MIRAI_SCAN_CSV
    names = feature_names()
    rng = np.random.default_rng(seed)

    n_attack = int(n_calib * attack_fraction)
    n_benign = n_calib - n_attack

    block = []
    for v in _rows(BENIGN_CSV, n_benign, benign_offset):
        block.append(_record(v, names, 0, BENIGN_REPUTATION, "calib_benign"))
    for v in _rows(attack_csv, n_attack, attack_offset):
        block.append(_record(v, names, 1, ATTACK_REPUTATION, "calib_attack"))

    idx = rng.permutation(len(block))
    return [block[i] for i in idx]


# --- Scenario 1: genuine drift -------------------------------------------

def genuine_drift_stream(n_calib=400, n_pre=400, n_drift=600, n_post=400,
                         benign_b_csv=None, seed=42):
    """
    Scenario 1 — legitimate behavioural drift to a NEW BENIGN operating point
    (e.g. a firmware update changes polling behaviour). Reputation stays clean
    (1.0) throughout: this is traffic the gate SHOULD eventually authorize.

    IMPORTANT — why this needs a second benign source
    -------------------------------------------------
    The README originally suggested splitting benign_traffic.csv in half. That
    was measured and it does not work: the file is stationary. Standardised
    distance between decile block 1 and block 10 is 0.16, while *consecutive*
    blocks wander 0.30-0.74 — i.e. the endpoint separation is smaller than the
    round-to-round noise. There is no directional drift to detect, ADWIN never
    fires, no update is ever proposed, and all four arms produce identical
    lines. A null result, not a demo.

    So the post-drift operating point must come from a genuinely different
    benign distribution. `benign_b_csv` should be the benign_traffic.csv of a
    SECOND device (same-type is the cleanest story). Both endpoints are then
    real, benign, clean-reputation traffic, and the drift between them is a
    real distribution shift rather than an invented one.

    If benign_b_csv is None this raises, rather than silently producing the
    null result described above.
    """
    if benign_b_csv is None:
        raise ValueError(
            "genuine_drift_stream needs benign_b_csv (a SECOND device's "
            "benign_traffic.csv).\n"
            "Splitting Danmini's own benign file does not work — measured "
            "block1-vs-block10 distance is 0.16 sigma against consecutive-block "
            "noise of 0.30-0.74, so ADWIN never fires and every arm is identical.\n"
            "See scripts/fetch_second_device.py, or pass a path explicitly."
        )

    names = feature_names()
    stream = list(calibration_block(n_calib, seed=seed))

    # pre-drift: device A's benign steady state
    off = (n_calib - int(n_calib * 0.3))
    for v in _rows(BENIGN_CSV, n_pre, off):
        stream.append(_record(v, names, 0, BENIGN_REPUTATION, "pre_drift"))

    # drift: gradual traversal from device A's benign point to device B's
    a_rows = _rows(BENIGN_CSV, n_drift, off + n_pre)
    b_rows = _rows(benign_b_csv, n_drift, 0)
    for i in range(n_drift):
        alpha = i / max(1, n_drift - 1)
        v = _interp(a_rows[i], b_rows[i], alpha)
        stream.append(_record(v, names, 0, BENIGN_REPUTATION, "genuine_drift"))

    # post-drift: settled at device B's benign operating point
    for v in _rows(benign_b_csv, n_post, n_drift):
        stream.append(_record(v, names, 0, BENIGN_REPUTATION, "post_drift"))

    return stream


# --- Scenario 2: slow poisoning ------------------------------------------

def slow_poisoning_stream(n_calib=400, n_pre=400, n_poison=600, n_post=400,
                          n_recovery_check=200, attack_csv=None, seed=42):
    """
    Scenario 2 — slow adversarial concept drift.

    A patient white-box attacker knows the IDS continually adapts, and paces
    real Mirai traffic so its feature values migrate slowly toward the benign
    region, trying to get the IDS to "learn" that attack traffic is normal.

    Crucially the destination is still C2 infrastructure throughout, so the
    out-of-band reputation stays bad (0.05) even once the traffic-derived
    features look benign. That gap is the entire point of the mechanism.

    Real N-BaIoT attack files are not temporally paced to be slow, so the
    pacing is constructed here (README step 4a) by interpolating real attack
    rows toward real benign rows in signed-log space — every endpoint is a
    genuine captured packet-window, only the schedule is imposed.

    Phases mirror synth_stream.slow_poisoning_stream exactly.
    """
    attack_csv = attack_csv or MIRAI_SCAN_CSV
    names = feature_names()
    stream = list(calibration_block(n_calib, attack_csv=attack_csv, seed=seed))

    b_off = n_calib - int(n_calib * 0.3)
    a_off = int(n_calib * 0.3)

    # pre-attack: clean benign steady state
    for v in _rows(BENIGN_CSV, n_pre, b_off):
        stream.append(_record(v, names, 0, BENIGN_REPUTATION, "pre_attack"))

    # slow poisoning: real attack rows walked toward real benign rows
    atk = _rows(attack_csv, n_poison, a_off)
    ben = _rows(BENIGN_CSV, n_poison, b_off + n_pre)
    for i in range(n_poison):
        alpha = i / max(1, n_poison - 1)
        v = _interp(atk[i], ben[i], alpha)
        stream.append(_record(v, names, 1, ATTACK_REPUTATION, "slow_poison"))

    # post-poison: attacker holds traffic in the benign region.
    # Ground truth is STILL attack — these are attack rows fully mapped onto
    # the benign manifold, which is what makes them undetectable from traffic
    # alone and detectable only via the out-of-band channel.
    atk_p = _rows(attack_csv, n_post, a_off + n_poison)
    ben_p = _rows(BENIGN_CSV, n_post, b_off + n_pre + n_poison)
    for i in range(n_post):
        v = _interp(atk_p[i], ben_p[i], 1.0)
        stream.append(_record(v, names, 1, ATTACK_REPUTATION, "post_poison"))

    # recovery check: an OBVIOUS, full-strength repeat of the attack the model
    # was calibrated on. If poisoning corrupted the model's notion of normal it
    # will miss this too — that lasting damage, not just the missed poisoning
    # window, is the real cost of an unauthorized update.
    for v in _rows(attack_csv, n_recovery_check, a_off + n_poison + n_post):
        stream.append(_record(v, names, 1, ATTACK_REPUTATION, "recovery_check"))

    return stream


# --- generic loader (kept from the original stub) -------------------------

def load_device_stream(benign_csv, attack_csvs, sample_n=None):
    """Plain benign-then-attack concatenation. Kept for ad-hoc inspection."""
    stream = []
    benign_df = pd.read_csv(benign_csv)
    if sample_n:
        benign_df = benign_df.sample(min(sample_n, len(benign_df)), random_state=42)
    for _, row in benign_df.iterrows():
        stream.append((row.to_dict(), 0, {"reputation": BENIGN_REPUTATION, "source": "benign"}))

    for path in attack_csvs:
        attack_df = pd.read_csv(path)
        if sample_n:
            attack_df = attack_df.sample(min(sample_n, len(attack_df)), random_state=42)
        for _, row in attack_df.iterrows():
            stream.append((row.to_dict(), 1, {"reputation": ATTACK_REPUTATION, "source": path}))
    return stream
