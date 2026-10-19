"""
Feature catalogue: what every input feature means, for both real datasets.

N-BaIoT (Meidan et al. 2018) ships 115 statistics per packet, computed by the
dataset authors with damped (exponentially decaying) incremental statistics,
the same extractor as Kitsune (Mirsky et al. 2018). A feature name has three
parts, e.g. HH_jit_L0.1_variance:

    <aggregation>_<decay window>_<statistic>

  aggregation  what the packets are grouped by
     MI_dir    source MAC + IP          (the sending host, by link + network id)
     H         source IP                (the sending host)
     HH        source IP -> dest IP     (the channel between two hosts)
     HH_jit    source IP -> dest IP     (the channel's inter-packet TIME, jitter)
     HpHp      src IP:port -> dst IP:port (the socket)
  decay window lambda in {5, 3, 1, 0.1, 0.01}: how fast old packets are
     forgotten, i.e. a time window of roughly 100 ms, 500 ms, 1.5 s, 10 s, 1 min
  statistic
     weight      decayed packet count (how many packets recently)
     mean        mean packet size (jitter for HH_jit)
     variance / std   spread of packet size (jitter for HH_jit)
     magnitude   sqrt(mean_out^2 + mean_in^2): size of traffic both ways
     radius      sqrt(var_out^2 + var_in^2): spread both ways
     covariance  covariance of outbound and inbound sizes
     pcc         Pearson correlation of outbound and inbound sizes

  1-D groups (MI_dir, H, HH_jit) have weight/mean/variance: 3 x 5 = 15 each.
  2-D groups (HH, HpHp) add magnitude/radius/covariance/pcc: 7 x 5 = 35 each.
  15 + 15 + 35 + 15 + 35 = 115.

IoT-23 (Garcia et al. 2020) records are Zeek connection logs, one row per
flow. data/load_iot23.py turns each into 21 features (below).
"""
import re

NBAIOT_GROUPS = {
    "MI_dir": "source MAC+IP (sending host)",
    "H": "source IP (sending host)",
    "HH": "source->destination IP channel",
    "HH_jit": "channel inter-packet jitter",
    "HpHp": "source->destination socket (IP:port)",
}
NBAIOT_WINDOWS = {"L5": "~100 ms", "L3": "~500 ms", "L1": "~1.5 s", "L0.1": "~10 s",
                  "L0.01": "~1 min"}
NBAIOT_STATS = {
    "weight": "decayed packet count",
    "mean": "mean packet size",
    "variance": "packet-size variance",
    "std": "packet-size standard deviation",
    "magnitude": "sqrt(mean_out^2 + mean_in^2)",
    "radius": "sqrt(var_out^2 + var_in^2)",
    "covariance": "covariance of outbound/inbound sizes",
    "pcc": "correlation of outbound/inbound sizes",
}
JITTER_STATS = {"weight": "decayed packet count", "mean": "mean inter-packet time",
                "variance": "inter-packet time variance"}
_NAME = re.compile(r"^(MI_dir|HH_jit|HpHp|HH|H)_(L[0-9.]+)_(\w+)$")

IOT23_FEATURES = {
    "duration": "flow duration in seconds",
    "orig_bytes": "payload bytes sent by the originator",
    "resp_bytes": "payload bytes sent by the responder",
    "missed_bytes": "bytes Zeek missed (packet loss)",
    "orig_pkts": "packets sent by the originator",
    "orig_ip_bytes": "IP-level bytes sent by the originator",
    "resp_pkts": "packets sent by the responder",
    "resp_ip_bytes": "IP-level bytes sent by the responder",
    "proto_tcp": "one-hot: transport protocol TCP",
    "proto_udp": "one-hot: transport protocol UDP",
    "proto_icmp": "one-hot: transport protocol ICMP",
    "state_S0": "one-hot: SYN seen, no reply (scan / unanswered)",
    "state_S1": "one-hot: established, not terminated",
    "state_SF": "one-hot: normal establish and termination",
    "state_REJ": "one-hot: connection attempt rejected",
    "state_RSTO": "one-hot: established, originator aborted (RST)",
    "state_RSTR": "one-hot: responder sent RST",
    "state_RSTOS0": "one-hot: originator SYN then RST, no SYN-ACK",
    "state_SH": "one-hot: originator SYN then FIN, no SYN-ACK",
    "state_SHR": "one-hot: responder SYN-ACK then FIN, no SYN",
    "state_OTH": "one-hot: no SYN seen, midstream traffic",
}


def parse_nbaiot_name(name):
    """'HH_jit_L0.1_variance' -> ('HH_jit', 'L0.1', 'variance'); None if not N-BaIoT."""
    m = _NAME.match(name)
    return m.groups() if m else None


def describe(name):
    """One-line plain-English meaning of any feature name of either dataset."""
    if name in IOT23_FEATURES:
        return IOT23_FEATURES[name]
    parts = parse_nbaiot_name(name)
    if parts is None:
        return "synthetic feature" if name.startswith("f") else "unknown feature"
    group, window, stat = parts
    what = (JITTER_STATS if group == "HH_jit" else NBAIOT_STATS).get(stat, stat)
    return f"{what}, {NBAIOT_GROUPS[group]}, window {NBAIOT_WINDOWS.get(window, window)}"


def catalogue(names):
    """[(name, group, window, statistic, meaning)] for a list of feature names."""
    rows = []
    for n in names:
        p = parse_nbaiot_name(n)
        group, window, stat = p if p else ("flow" if n in IOT23_FEATURES else "-", "-", "-")
        rows.append((n, group, window, stat, describe(n)))
    return rows
