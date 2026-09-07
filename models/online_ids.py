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
from river import linear_model, preprocessing, metrics, drift


class OnlineIDS:
    def __init__(self, drift_detector=True):
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
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
