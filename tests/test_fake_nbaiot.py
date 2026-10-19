import os

import numpy as np
import pandas as pd
import pytest

from data import fake_nbaiot as fk
from data import load_nbaiot as nb

FILES = [("Danmini_Doorbell", "benign_traffic.csv"),
         ("Danmini_Doorbell", "mirai_attacks/scan.csv"),
         ("Danmini_Doorbell", "gafgyt_attacks/combo.csv"),
         ("Ennio_Doorbell", "benign_traffic.csv")]


def _df(root, device, rel):
    return pd.read_csv(os.path.join(root, device, rel))


def test_115_real_column_names():
    names = fk.feature_names()
    assert len(names) == 115 == len(set(names))
    assert names[0] == "MI_dir_L5_weight"
    assert names[-1] == "HpHp_L0.01_pcc"
    assert sum(n.endswith(("_covariance", "_pcc")) for n in names) == 20


@pytest.mark.parametrize("device,rel", FILES)
def test_layout_shape_and_finiteness(fake_root, device, rel):
    df = _df(fake_root, device, rel)
    assert list(df.columns) == fk.feature_names()
    assert len(df) == 2500
    assert np.isfinite(df.to_numpy()).all()


def test_marker_file_says_fake(fake_root):
    with open(os.path.join(fake_root, fk.MARKER)) as f:
        assert "NOT REAL DATA" in f.read()


def test_signed_and_bounded_columns(fake_root):
    df = _df(fake_root, *FILES[0])
    pcc = df[[c for c in df.columns if c.endswith("_pcc")]].to_numpy()
    assert pcc.min() >= -1 and pcc.max() <= 1
    signed = df[[c for c in df.columns if c.endswith(("_covariance", "_pcc"))]].to_numpy()
    assert (signed < 0).any()
    unsigned = df[[c for c in df.columns if not c.endswith(("_covariance", "_pcc"))]]
    assert (unsigned.to_numpy() >= 0).all()


def test_real_dynamic_range_is_mimicked(fake_root):
    df = _df(fake_root, *FILES[0])
    jit_var = df[[c for c in df.columns if c.startswith("HH_jit") and c.endswith("variance")]]
    assert jit_var.to_numpy().min() > 1e15          # the ~1e17 outlier columns
    assert df.abs().to_numpy().max() / df["MI_dir_L5_weight"].abs().median() > 1e12


def test_attack_and_second_device_differ_from_benign(fake_root):
    def centre(device, rel):
        d = _df(fake_root, device, rel).to_numpy()
        return np.sign(d) * np.log1p(np.abs(d))
    benign = centre(*FILES[0])
    noise = np.linalg.norm(benign[:1250].mean(0) - benign[1250:].mean(0))
    for f in FILES[1:]:
        assert np.linalg.norm(centre(*f).mean(0) - benign.mean(0)) > 10 * noise


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fk.generate(str(a), rows=50, seed=3)
    fk.generate(str(b), rows=50, seed=3)
    for f in FILES:
        pd.testing.assert_frame_equal(_df(a, *f), _df(b, *f))


def test_ensure_does_not_regenerate(tmp_path):
    fk.ensure(str(tmp_path), rows=30, seed=1)
    path = tmp_path / "Danmini_Doorbell" / "benign_traffic.csv"
    mtime = path.stat().st_mtime_ns
    fk.ensure(str(tmp_path), rows=30, seed=2)
    assert path.stat().st_mtime_ns == mtime


# --- the real-data loader, driven by fake files ---------------------------

def test_loader_slow_poisoning_on_fake_files(fake_root):
    stream = nb.slow_poisoning_stream(root=fake_root)
    assert len(stream) == 400 + 400 + 600 + 400 + 200
    x, y, meta = stream[-1]
    assert len(x) == 115 and y == 1 and meta["source"] == "recovery_check"
    assert {m["source"] for _, _, m in stream} == {
        "calib_benign", "calib_attack", "pre_attack", "slow_poison", "post_poison",
        "recovery_check"}
    labels = {m["source"]: y for _, y, m in stream}
    assert labels["post_poison"] == 1 and labels["pre_attack"] == 0


def test_loader_genuine_drift_on_fake_files(fake_root):
    b = os.path.join(fake_root, "Ennio_Doorbell", "benign_traffic.csv")
    stream = nb.genuine_drift_stream(benign_b_csv=b, root=fake_root)
    assert len(stream) == 400 + 400 + 600 + 400
    assert all(y == 0 for _, y, m in stream if m["source"] != "calib_attack")
    assert all(m["reputation"] == 1.0 for _, _, m in stream if m["source"] != "calib_attack")


def test_loader_still_refuses_genuine_drift_without_second_device(fake_root):
    with pytest.raises(ValueError, match="SECOND device"):
        nb.genuine_drift_stream(root=fake_root)
