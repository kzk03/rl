import math
from datetime import datetime, timedelta

import numpy as np

from gerrit_retention.prediction.irl_feature_adapter import IRLFeatureAdapter


class DummyIRL:
    def __init__(self, p: float = 0.7):
        self._p = p

    def predict_proba(self, states):
        return [self._p for _ in states]


def test_compute_features_shape_and_values():
    adapter = IRLFeatureAdapter(irl_model=DummyIRL(0.8))
    developer = {"days_since_last_activity": 5}
    context = {"context_date": datetime.now()}

    feats, names = adapter.compute_features(developer, context)

    assert isinstance(names, list) and len(names) == 3
    assert feats.shape == (3,)

    p, logit, ent = feats.tolist()
    # Probability is in (0,1)
    assert 0 < p < 1
    # Logit and entropy finite
    assert math.isfinite(logit)
    assert 0 <= ent <= 1.0  # entropy of Bernoulli max is <= 1 for natural log
