"""
Synthetic stream generator — stands in for real N-BaIoT CSVs so the pipeline
can be built and tested TODAY without a dataset download.

Schema: N_FEATURES numeric features per record (real N-BaIoT has 115;
we use 20 here for a fast demo — swap in load_nbaiot.py for the real thing,
same downstream interface: yields (features_dict, true_label, meta_dict)).

Ground truth generative process (all parameters below are made up for the
demo and must be replaced by real data statistics before Review-2 — this
file exists ONLY to unblock pipeline development, never cite its numbers
as results).
"""
import numpy as np

N_FEATURES = 20
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]

rng = np.random.default_rng(42)

BENIGN_MEAN = rng.normal(0, 1, N_FEATURES)
ATTACK_MEAN = BENIGN_MEAN + rng.normal(4, 0.5, N_FEATURES)  # well-separated
NOISE_STD = 1.0


def _sample(mean_vec, n=1):
    return rng.normal(mean_vec, NOISE_STD, size=(n, N_FEATURES))


def make_record(mean_vec, label, reputation, src="synthetic"):
    x = _sample(mean_vec, 1)[0]
    feats = {name: float(v) for name, v in zip(FEATURE_NAMES, x)}
    meta = {"reputation": reputation, "source": src}
    return feats, label, meta


def _calibration_block(n_calib, attack_fraction=0.3):
    """
    Labelled bootstrap set: a real IDS is never trained from zero, it starts
    from a historical labelled dataset that already contains known-attack
    examples (e.g. earlier Mirai variants) alongside benign traffic. Without
    this, the model has no concept of "attack" at all and every scenario
    below is meaningless.
    """
    block = []
    n_attack = int(n_calib * attack_fraction)
    n_benign = n_calib - n_attack
    for _ in range(n_benign):
        block.append(make_record(BENIGN_MEAN, 0, reputation=1.0, src="calib_benign"))
    for _ in range(n_attack):
        block.append(make_record(ATTACK_MEAN, 1, reputation=0.05, src="calib_attack"))
    rng.shuffle(block)
    return block


def genuine_drift_stream(n_calib=400, n_pre=400, n_drift=600, n_post=400):
    """
    Scenario 1 — legitimate behavioural drift.
    A benign device's traffic pattern gradually shifts to a NEW benign
    operating point (e.g. firmware update changes polling behaviour).
    Destination reputation stays clean throughout (reputation=1.0).
    """
    stream = []
    new_benign_mean = BENIGN_MEAN + rng.normal(2.5, 0.3, N_FEATURES)

    stream.extend(_calibration_block(n_calib))
    for _ in range(n_pre):
        stream.append(make_record(BENIGN_MEAN, 0, reputation=1.0, src="pre_drift"))
    for i in range(n_drift):
        alpha = i / n_drift
        mean_i = (1 - alpha) * BENIGN_MEAN + alpha * new_benign_mean
        stream.append(make_record(mean_i, 0, reputation=1.0, src="genuine_drift"))
    for _ in range(n_post):
        stream.append(make_record(new_benign_mean, 0, reputation=1.0, src="post_drift"))
    return stream


def slow_poisoning_stream(n_calib=400, n_pre=400, n_poison=600, n_post=400, n_recovery_check=200):
    """
    Scenario 2 — slow adversarial concept drift.
    A patient, white-box attacker knows the IDS continually adapts, and
    paces malicious (Mirai-like) traffic so its FEATURE VALUES drift slowly
    toward the benign region — trying to get the IDS to "learn" that this
    attack traffic is normal.
    Crucially: the destination is still C2 infrastructure the whole time,
    so the out-of-band reputation signal stays bad (reputation=0.05)
    even as the traffic-derived features start to look benign.
    """
    stream = []
    stream.extend(_calibration_block(n_calib))
    for _ in range(n_pre):
        stream.append(make_record(BENIGN_MEAN, 0, reputation=1.0, src="pre_attack"))
    for i in range(n_poison):
        alpha = i / n_poison
        mean_i = (1 - alpha) * ATTACK_MEAN + alpha * BENIGN_MEAN
        stream.append(make_record(mean_i, 1, reputation=0.05, src="slow_poison"))
    for _ in range(n_post):
        # attacker keeps traffic looking benign; ground truth is still an attack
        stream.append(make_record(BENIGN_MEAN, 1, reputation=0.05, src="post_poison"))
    for _ in range(n_recovery_check):
        # the payoff question: after the poisoning attempt, does the IDS still
        # catch an OBVIOUS, unmistakable repeat attack (full-strength, same
        # signature as calibration)? If poisoning corrupted what the model
        # believes "normal" looks like, it may miss this too — that lasting
        # damage, not just the missed slow-poison window, is the real cost
        # of an unauthorized update.
        stream.append(make_record(ATTACK_MEAN, 1, reputation=0.05, src="recovery_check"))
    return stream
