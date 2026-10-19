"""
Imperfect out-of-band reputation channel.

The stock streams key meta["reputation"] directly to ground-truth origin
(benign 1.0, attack 0.05) — an oracle. A real threat-intel / C2-cost feed is
noisy, late and sometimes wrong. This wraps any stream and degrades the
channel in three independent, separately controllable ways:

  sigma      Gaussian measurement noise on every record, clipped to [0, 1]
  flip_rate  per-record probability the feed reports the WRONG class
             (benign <-> attack value swapped: a stale or poisoned blocklist)
  lag        the feed reports record i-lag's reputation for record i
             (threat intel arriving late); the first `lag` records keep
             their own value

Only meta["reputation"] changes; features, labels and phases are untouched,
so the comparison against the oracle run stays controlled. The original is
kept as meta["reputation_true"].

The gate averages reputation over a 20-record window, so per-record noise and
flips partly wash out; `lag` and high flip rates are the ones that can move a
whole window across the veto threshold.
"""
from dataclasses import dataclass

import numpy as np

BENIGN_REPUTATION = 1.0
ATTACK_REPUTATION = 0.05


@dataclass(frozen=True)
class ReputationNoise:
    sigma: float = 0.0
    flip_rate: float = 0.0
    lag: int = 0
    seed: int = 0

    def __post_init__(self):
        if self.sigma < 0 or not 0 <= self.flip_rate <= 1 or self.lag < 0:
            raise ValueError(f"invalid reputation noise {self}")

    @property
    def is_oracle(self):
        return self.sigma == 0 and self.flip_rate == 0 and self.lag == 0


def apply_reputation_noise(stream, noise: ReputationNoise):
    """Return a new stream with a degraded reputation channel. Input is not mutated."""
    if noise.is_oracle:
        return list(stream)
    rng = np.random.default_rng(noise.seed)
    true_rep = np.array([m["reputation"] for _, _, m in stream], dtype=float)

    rep = true_rep.copy()
    if noise.lag:
        rep[noise.lag:] = true_rep[:-noise.lag]
    if noise.flip_rate:
        flip = rng.random(len(rep)) < noise.flip_rate
        rep = np.where(flip, BENIGN_REPUTATION + ATTACK_REPUTATION - rep, rep)
    if noise.sigma:
        rep = rep + rng.normal(0, noise.sigma, len(rep))
    rep = np.clip(rep, 0.0, 1.0)

    out = []
    for (x, y, meta), r, t in zip(stream, rep, true_rep):
        out.append((x, y, {**meta, "reputation": float(r), "reputation_true": float(t)}))
    return out
