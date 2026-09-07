"""
Real N-BaIoT loader — fill this in once the dataset is downloaded locally.

Download (do this on YOUR machine, not in a sandboxed environment):
  https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
  -> pick ONE device to start (e.g. Danmini_Doorbell) to keep it small:
       Danmini_Doorbell/benign_traffic.csv
       Danmini_Doorbell/gafgyt_attacks/combo.csv, .../scan.csv, ...
       Danmini_Doorbell/mirai_attacks/ack.csv, .../scan.csv, .../udp.csv, ...
  Each CSV already has the 115 pre-extracted numeric features as columns —
  no Zeek/pcap parsing needed. That's why N-BaIoT (one of your three
  originally-proposed datasets) is the fastest path to a REAL working demo.

This module's job: yield the exact same (features_dict, label, meta_dict)
tuples that synth_stream.py yields, so experiments/run_scenarios.py doesn't
care whether the data is real or synthetic.

meta["reputation"] is the one field with NO equivalent in N-BaIoT — it's your
out-of-band signal, and N-BaIoT has no destination IP/reputation info at all
(commit to labelling it CLEARLY as a simulated/proof-of-concept out-of-band
feed, exactly as the correction message told you to do — do not claim it's
a real threat-intel lookup).
Simple honest proxy: reputation = 1.0 for rows drawn from benign_traffic.csv,
reputation = 0.05 for rows drawn from *_attacks/*.csv. This is defensible in
viva as "we simulate the out-of-band channel using ground-truth traffic
origin, standing in for a real reputation/C2-cost feed we did not have time
to integrate by Review-2."
"""
import pandas as pd


def load_device_stream(benign_csv, attack_csvs, order="temporal_concat", sample_n=None):
    """
    benign_csv: path to benign_traffic.csv
    attack_csvs: list of paths to attack CSVs (mix mirai/gafgyt as needed)
    order: "temporal_concat" -> benign block then attack block (mimics a
           device that starts clean and gets infected — matches Scenario 2
           framing). For Scenario 1 (genuine drift) use two DIFFERENT
           benign-labelled slices (e.g. different devices of the same type,
           or split benign_traffic.csv in half) rather than an attack file.

    Returns: list of (features_dict, label, meta_dict)
    """
    stream = []

    benign_df = pd.read_csv(benign_csv)
    if sample_n:
        benign_df = benign_df.sample(min(sample_n, len(benign_df)), random_state=42)
    for _, row in benign_df.iterrows():
        feats = row.to_dict()
        stream.append((feats, 0, {"reputation": 1.0, "source": "benign"}))

    for path in attack_csvs:
        attack_df = pd.read_csv(path)
        if sample_n:
            attack_df = attack_df.sample(min(sample_n, len(attack_df)), random_state=42)
        for _, row in attack_df.iterrows():
            feats = row.to_dict()
            stream.append((feats, 1, {"reputation": 0.05, "source": path}))

    return stream
