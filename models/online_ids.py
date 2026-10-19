"""
Baseline online/continual-learning IDS, built on River.

Why River (decided for the 2-day Review-2 deadline):
- Purpose-built for one-sample-at-a-time streaming/continual learning —
  matches the project's actual threat model (a model that keeps adapting)
  far more directly than batch scikit-learn + manual partial_fit loops.
- Ships ADWIN out of the box for drift *detection*, which is one of the
  pipeline stages already named in the architecture — saves writing and
  debugging a detector from scratch.
- Pure Python, tiny dependency footprint, trains on commodity hardware in
  seconds-to-minutes even on the full N-BaIoT device CSVs — no GPU, no
  batch/epoch tuning. A PyTorch implementation would spend most of the
  two days on training-loop plumbing instead of the trust-gate logic,
  which is the actual research contribution and what Review-2 grades.
- Existing GitHub IDS+River repos (e.g. PWPAE) are useful as a reference
  for the River+ADWIN pattern (see code below) but are built around a
  different ensemble-drift-adaptation problem; adapting one to insert a
  custom mid-pipeline trust gate would cost more time than writing this
  ~80-line wrapper from scratch, and a self-written model is easier to
  defend line-by-line in viva.
"""
from river import drift, forest, linear_model, metrics, naive_bayes, preprocessing, tree

# Base learners the gate can wrap (run_scenario(base_model=...)). "lr" is the
# project's detector and the default; the others show the gate is not tied to it.
BASE_MODELS = {
    "lr": lambda: preprocessing.StandardScaler() | linear_model.LogisticRegression(),
    "hoeffding_tree": lambda: tree.HoeffdingTreeClassifier(),
    "hoeffding_adaptive_tree": lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=1),
    "arf": lambda: forest.ARFClassifier(n_models=5, seed=1),
    "naive_bayes": lambda: preprocessing.StandardScaler() | naive_bayes.GaussianNB(),
}
BASE_MODEL_NAMES = {
    "lr": "Logistic regression (ours)",
    "hoeffding_tree": "Hoeffding tree",
    "hoeffding_adaptive_tree": "Hoeffding adaptive tree",
    "arf": "Adaptive random forest",
    "naive_bayes": "Gaussian naive Bayes",
}


class OnlineIDS:
    def __init__(self, drift_detector=True, base_model="lr"):
        if base_model not in BASE_MODELS:
            raise ValueError(f"unknown base_model {base_model!r}; expected one of "
                             f"{sorted(BASE_MODELS)}")
        self.model = BASE_MODELS[base_model]()
        self.metric = metrics.Accuracy()
        self.f1 = metrics.F1()
        self.adwin = drift.ADWIN() if drift_detector else None
        self.n_seen = 0
        self.n_updates_applied = 0

    def predict_one(self, x):
        return self.model.predict_one(x) or 0

    def score_one(self, x, y_true):
        """Prequential step: predict first, THEN report correctness — never
        let the model see y before predicting on that sample."""
        y_pred = self.predict_one(x)
        self.metric.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.n_seen += 1
        if self.adwin is not None:
            self.adwin.update(int(y_pred != y_true))
        return y_pred

    def learn_one(self, x, y_true):
        self.model.learn_one(x, y_true)
        self.n_updates_applied += 1

    def drift_detected(self):
        return self.adwin is not None and self.adwin.drift_detected

    def learn_batch(self, records):
        """Apply an AUTHORIZED batch of (x, y) to the model."""
        for x, y in records:
            self.learn_one(x, y)


class EWCOnlineIDS(OnlineIDS):
    """
    Elastic Weight Consolidation baseline: learns from EVERY drift-triggered
    proposal (like the ungated arm), but each SGD step is followed by a step
    on the EWC penalty

        (lambda / 2) * sum_i F_i * (w_i - w*_i)^2

    where w* are the logistic-regression weights after calibration and F is
    the diagonal Fisher information of the calibration data, normalised to
    max 1. Important weights are pulled back toward what calibration learned;
    unimportant ones move freely.

    This is the standard defence against FORGETTING. It has no notion of
    whether an update is authorized, which is exactly why it is the right
    baseline: it shows how much of the gate's benefit is just "don't forget".
    Limitation, stated plainly: only the LR weights are anchored; the
    StandardScaler in front of them keeps updating its running statistics.
    """

    def __init__(self, ewc_lambda=50.0, drift_detector=True):
        super().__init__(drift_detector=drift_detector)
        self.ewc_lambda = ewc_lambda
        self.fisher = None
        self.anchor = None

    @property
    def _scaler(self):
        return self.model["StandardScaler"]

    @property
    def _lr(self):
        return self.model["LogisticRegression"]

    def consolidate(self, xs):
        """Freeze the current weights as w* and estimate F on calibration inputs."""
        fisher, fisher_b = {}, 0.0
        for x in xs:
            z = self._scaler.transform_one(x)
            p = self._lr.predict_proba_one(z).get(True, 0.0)
            g = p * (1.0 - p)
            for k, v in z.items():
                fisher[k] = fisher.get(k, 0.0) + g * v * v
            fisher_b += g
        top = max(list(fisher.values()) + [fisher_b, 1e-12])
        self.fisher = {k: v / top for k, v in fisher.items()}
        self.fisher_intercept = fisher_b / top
        self.anchor = dict(self._lr._weights)
        self.anchor_intercept = self._lr.intercept

    def learn_one(self, x, y_true):
        super().learn_one(x, y_true)
        if self.anchor is None:
            return
        lr = self._lr
        step = lr.optimizer.learning_rate * self.ewc_lambda
        w = lr._weights
        for k, w_star in self.anchor.items():
            c = min(1.0, step * self.fisher.get(k, 0.0))
            w[k] = w[k] - c * (w[k] - w_star)
        c = min(1.0, step * self.fisher_intercept)
        lr.intercept = lr.intercept - c * (lr.intercept - self.anchor_intercept)
