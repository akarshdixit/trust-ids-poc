
"""
Calibrated feature space for evidence extraction.

WHY THIS MODULE EXISTS (do not delete it as "just scaling"):
------------------------------------------------------------
The three evidence signals in extract.py are all GEOMETRIC — they measure
cosine angles and Euclidean norms between window mean vectors. Geometry is
only meaningful if every feature contributes on a comparable scale.

On the synthetic stream that was free: all 20 features were N(0, 1).
On real N-BaIoT it is catastrophically false. Measured on
Danmini_Doorbell/benign_traffic.csv (115 features):

    feature std: min 6.6e-03   median 1.8e+01   max 3.2e+16
    feature max: min 4.9e-01   median 3.6e+02   max 5.7e+17

That is ~19 orders of magnitude. Feeding raw vectors to extract.py gives:

    temporal_stability  = 1.7e-17   (1/(1+||delta||), ||delta|| = 5.8e16)
    -> the 0.3 temporal weight is dead budget in EVERY round, for every arm

    attribution_consistency: five HH_jit_*_variance features account for
    100.0% of the shift vector's squared norm (71.8 / 11.6 / 6.6 / 5.4 /
    4.6 %), so the "cosine similarity across 115 features" is really a
    one-feature measurement.

Net effect: a genuinely BENIGN window scored trust = 0.5295 against the
0.55 threshold, i.e. it was REFUSED. With nothing ever authorized, all
four arms degenerate to the static arm and the experiment shows nothing.

The fix is to compute evidence in a space where the geometry is honest:

    1. signed log  ->  sign(x) * log1p(|x|)
       Compresses the 19-decade dynamic range. Signed rather than log1p(|x|)
       because 20 of the 115 columns (HH_*_covariance, HH_*_pcc, and the
       HpHp equivalents) take negative values -- a Pearson correlation of
       -0.9 and +0.9 are opposite behaviours and must not be folded together.

    2. z-score using mean/std FIT ON CALIBRATION BENIGN DATA ONLY.
       Fitting on the live window would let the attacker move the reference
       frame, which is exactly the influence the gate exists to deny. This
       is a trust boundary, not a preprocessing convenience.

Same window after the transform: temporal 0.2572, topology 0.8292,
attribution 0.9097 -> trust 0.6898, AUTHORIZED. A raw Mirai window scores
0.43 and is refused by the traffic-only gate on evidence alone.

NOTE: this is deliberately NOT the scaler inside OnlineIDS. River's
preprocessing.StandardScaler lives inside the model pipeline and adapts
online as the model sees data; this one is frozen at calibration time
precisely so the evidence layer has a fixed, trusted frame of reference.
"""
import numpy as np


def signed_log(a):
    """sign(x) * log1p(|x|) — compresses magnitude, preserves direction."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


class FeatureSpace:
    """
    Frozen, calibration-fitted transform applied to every vector before it
    reaches evidence/extract.py.

    Usage:
        fs = FeatureSpace().fit(calibration_benign_matrix)
        z  = fs.transform(window_matrix)
    """

    def __init__(self, use_signed_log=True):
        self.use_signed_log = use_signed_log
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None

    def _pre(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return signed_log(X) if self.use_signed_log else X

    def fit(self, X):
        """Fit on TRUSTED calibration data only. Never refit on live windows."""
        Z = self._pre(X)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        # Constant columns carry no information; map them to exactly 0 rather
        # than dividing by ~0 and manufacturing enormous spurious values.
        self.std_[self.std_ < 1e-9] = 1.0
        self.n_features_ = Z.shape[1]
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("FeatureSpace.transform() called before fit()")
        Z = self._pre(X)
        if Z.shape[1] != self.n_features_:
            raise ValueError(
                f"FeatureSpace fit on {self.n_features_} features, got {Z.shape[1]}"
            )
        return (Z - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
