import datetime as dt
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cs2edge.model import MatchPredictor
from cs2edge.predict import Analyzer
from cs2edge.sources import simulate_history


class GBDTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("sklearn") is None:
            if os.environ.get("CS2EDGE_REQUIRE_GBDT") == "1":
                raise RuntimeError("scikit-learn est requis pour cette verification")
            raise unittest.SkipTest("option GBDT non installee")
        rng = np.random.default_rng(91)
        cls.X = rng.normal(size=(600, 4))
        y = (cls.X[:, 0] + rng.normal(size=600)*.7 > 0).astype(float)
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(600)]
        cls.model = MatchPredictor(use_gbdt=True).fit(cls.X, y, dates, ("a", "b", "c", "d"))
        if cls.model.gbdt is None:
            raise AssertionError("le GBDT doit reellement etre entraine")

    def test_actual_ensemble_is_symmetric(self):
        np.testing.assert_allclose(self.model.predict_proba(self.X) +
                                   self.model.predict_proba(-self.X), 1, atol=1e-12)

    def test_ensemble_save_load_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            loaded = MatchPredictor.load(path)
            self.assertIsNotNone(loaded.gbdt)
            np.testing.assert_allclose(loaded.predict_proba(self.X),
                                       self.model.predict_proba(self.X), atol=1e-12)

    def test_missing_ensemble_component_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            path.with_suffix(".gbdt.pkl").unlink()
            with self.assertRaisesRegex(ValueError, "GBDT.*manquant"):
                MatchPredictor.load(path)

    def test_mismatched_ensemble_component_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            blob = path.with_suffix(".gbdt.pkl")
            blob.write_bytes(blob.read_bytes() + b"different artifact")
            with self.assertRaisesRegex(ValueError, "GBDT incoherent"):
                MatchPredictor.load(path)

    def test_complete_analyzer_snapshot_keeps_the_actual_ensemble(self):
        maps = simulate_history(n_teams=8, years=1, seed=11, start=dt.date(2023, 1, 1))
        an = Analyzer().load_maps(maps).train(use_gbdt=True)
        self.assertIsNotNone(an.match_model.gbdt)
        on = max(m.date for m in maps) + dt.timedelta(days=1)
        a, b = an.known_teams()[:2]
        expected = an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            an.save(path)
            with patch.object(MatchPredictor, "fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                self.assertIsNotNone(restored.match_model.gbdt)
                self.assertIsNone(restored.map_model.gbdt)
                self.assertEqual(restored.predict(a, b, on=on), expected)
            for version in (None, "incompatible"):
                with patch("cs2edge.snapshot._sklearn_version", return_value=version):
                    with self.assertRaisesRegex(ValueError, "meme version de scikit-learn"):
                        Analyzer.load(path)


if __name__ == "__main__":
    unittest.main()
