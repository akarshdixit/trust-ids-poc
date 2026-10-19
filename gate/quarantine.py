"""
Quarantined authorization: provisional updates, sequential out-of-band
evidence, and rollback.

The one-shot gate (trust_gate.py) decides each update at the moment it is
proposed, from one 20-record window of reputation. Two findings on real data
show why that is brittle:

  * an imperfect feed (20% wrong records) pushes some attack windows' mean
    reputation above the fixed 0.3 veto, and those updates are kept forever;
  * a LATE feed (threat intel arriving after the traffic) shows benign
    readings exactly when the poisoned update is decided.

Here an update that passes the traffic evidence is applied PROVISIONALLY,
with a checkpoint of the model taken just before it. The update stays in
quarantine for `quarantine_rounds` rounds while out-of-band evidence about the
drift episode keeps accumulating in a Wald sequential probability ratio test.
If the test condemns the episode, every still-provisional update from it is
reverted (the model is restored to the earliest checkpoint and the
provisional updates of OTHER episodes are replayed on top). Updates that
survive quarantine are committed and can no longer be reverted.

Because a wrong acceptance is now recoverable, the traffic-evidence threshold
for a provisional update can be lower than the one-shot gate's.

Feed model behind the SPRT, stated plainly: each record's reputation is read
as a vote (benign if > 0.5), and the feed is assumed to be wrong on at most a
fraction `feed_error` of records. Under "episode is benign" a benign vote has
probability 1 - feed_error, under "episode is an attack" it has feed_error.
Wald's bounds give the requested false-condemn (alpha) and miss (beta) rates
when the assumption holds. The statistic is clamped to [lower, upper] so a
long benign prefix (a lagged feed) cannot bank evidence an attack must first
burn off; reaching `lower` condemns the episode for good.
"""
import copy
import math
from dataclasses import dataclass, field


class SequentialReputationTest:
    def __init__(self, feed_error=0.3, alpha=0.01, beta=0.01):
        if not 0 < feed_error < 0.5:
            raise ValueError(f"feed_error must be in (0, 0.5), got {feed_error}")
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError(f"alpha and beta must be in (0, 1), got {alpha}, {beta}")
        self.step = math.log((1 - feed_error) / feed_error)
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))

        # every vote moves the LLR by +-step, so track the net vote count exactly
        self.max_net = math.floor(self.upper / self.step)
        self.reset()

    def reset(self):
        self.net = 0  # benign votes minus attack votes, clamped at max_net
        self.condemned = False

    @property
    def llr(self):
        return self.net * self.step

    def update(self, reputations):
        """Add one window of per-record readings; returns the clamped LLR."""
        for r in reputations:
            if self.condemned:
                break
            self.net = min(self.net + (1 if r > 0.5 else -1), self.max_net)
            self.condemned = self.llr <= self.lower
        return self.llr


@dataclass
class ProvisionalUpdate:
    round: int
    episode: int
    batch: list
    window_mean: object
    checkpoint: object        # model state just BEFORE this update
    evidence_state: tuple     # (trusted_baseline, reference_direction) before it


@dataclass
class Quarantine:
    quarantine_rounds: int = 20
    pending: list = field(default_factory=list)
    n_committed: int = 0
    n_rolled_back: int = 0
    reverted_rounds: list = field(default_factory=list)  # rounds whose update was undone

    def add(self, round_idx, episode, batch, window_mean, ids, evidence_state):
        self.pending.append(ProvisionalUpdate(
            round_idx, episode, list(batch), window_mean.copy(),
            copy.deepcopy(ids.model), copy.deepcopy(evidence_state)))

    def commit_due(self, round_idx):
        keep = [u for u in self.pending if round_idx - u.round < self.quarantine_rounds]
        self.n_committed += len(self.pending) - len(keep)
        self.pending = keep

    def rollback(self, episode, ids, advance_evidence):
        """Revert every pending update of `episode`.

        advance_evidence(state, window_mean) -> state replays the evidence
        bookkeeping of a surviving update. Returns the restored evidence state,
        or None if nothing was pending for this episode.
        """
        idx = next((i for i, u in enumerate(self.pending) if u.episode == episode), None)
        if idx is None:
            return None
        first = self.pending[idx]
        ids.model = copy.deepcopy(first.checkpoint)
        state = copy.deepcopy(first.evidence_state)
        survivors = self.pending[:idx]
        for u in self.pending[idx:]:
            if u.episode == episode:
                self.n_rolled_back += 1
                self.reverted_rounds.append(u.round)
                continue
            u.checkpoint = copy.deepcopy(ids.model)
            u.evidence_state = copy.deepcopy(state)
            ids.learn_batch(u.batch)
            state = advance_evidence(state, u.window_mean)
            survivors.append(u)
        self.pending = survivors
        return state
