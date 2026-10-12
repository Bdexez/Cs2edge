import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cs2edge.model import MatchPredictor


class ModelIntegrityTests(unittest.TestCase):
    def test_invalid_model_metadata_is_rejected(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            original = json.loads(path.read_text())
            for changed in (dict(version=99), dict(scales=[0, 1]), dict(weights=[float("nan"), 1]),
                            dict(temperature=0), dict(resid=[1, 2]), dict(feature_names=["one"])):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**original, **changed}))
                    MatchPredictor.load(path)


if __name__ == "__main__":
    unittest.main()
