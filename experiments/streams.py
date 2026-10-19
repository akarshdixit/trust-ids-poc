"""
One place that turns (scenario, source, options) into a stream.

Used by experiments/run_scenarios.py, backend/server.py and app.py, so the
CLI, the API and the demo can never build a scenario differently.

Sources:
  synth  data/synth_stream.py      20 features, always available
  fake   data/fake_nbaiot.py       real N-BaIoT file format, generated locally
  real   data/nbaiot/ (UCI)        N-BaIoT, the dataset the report's results use
  iot23  data/iot23/ (Stratosphere) IoT-23 labelled Zeek flows: a second real dataset
"""
import os

from data.reputation import ReputationNoise, apply_reputation_noise

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ("slow_poisoning", "genuine_drift")
SOURCES = ("synth", "fake", "real", "iot23")
SECOND_DEVICE = "Ennio_Doorbell"


def build_stream(scenario, source="synth", seed=None, reputation_noise=None,
                 benign_b_csv=None):
    """Return (stream, feature_names).

    seed=None keeps each source's historic default (synth: continue the shared
    generator; fake/real: the loader's seed=42 calibration shuffle).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")

    if source == "synth":
        from data import synth_stream as ss
        fn = ss.slow_poisoning_stream if scenario == "slow_poisoning" else ss.genuine_drift_stream
        stream, names = fn(seed=seed), ss.FEATURE_NAMES
    elif source == "iot23":
        from data import load_iot23 as io
        fn = io.slow_poisoning_stream if scenario == "slow_poisoning" else io.genuine_drift_stream
        stream = fn(**({"seed": seed} if seed is not None else {}))
        names = io.feature_names()
    else:
        from data import load_nbaiot as nb
        if source == "fake":
            from data.fake_nbaiot import ensure
            root = ensure()
        else:
            root = nb.DEFAULT_ROOT
        kw = {"root": root}
        if seed is not None:
            kw["seed"] = seed
        if scenario == "slow_poisoning":
            stream = nb.slow_poisoning_stream(**kw)
        else:
            b = benign_b_csv or os.path.join(root, SECOND_DEVICE, "benign_traffic.csv")
            stream = nb.genuine_drift_stream(benign_b_csv=b, **kw)
        names = nb.feature_names(root=root)

    if reputation_noise is not None:
        if not isinstance(reputation_noise, ReputationNoise):
            reputation_noise = ReputationNoise(**reputation_noise)
        stream = apply_reputation_noise(stream, reputation_noise)
    return stream, names


def real_data_available():
    from data import load_nbaiot as nb
    return os.path.exists(nb.paths()["benign"])
