import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from cs2edge.predict import Analyzer
from cs2edge.ratings import EloConfig
from cs2edge.sources import simulate_history


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = simulate_history(n_teams=8, years=.5, seed=11, start=dt.date(2023, 1, 1))
        cls.an = Analyzer(EloConfig(k_base=14), warmup_maps=4, momentum=.1).load_maps(cls.maps)
        cls.an.train(use_gbdt=False, l2=3, calibration_frac=.25, recency_halflife=400)
        cls.on = max(m.date for m in cls.maps) + dt.timedelta(days=1)

    def test_round_trip_preserves_both_models_and_prediction_without_fitting(self):
        a, b = self.an.known_teams()[:2]
        expected = self.an.predict(a, b, on=self.on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            self.an.save(path)
            with patch.object(Analyzer, "train", side_effect=AssertionError("nouvel entrainement")), \
                    patch("cs2edge.model.MatchPredictor.fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                actual = restored.predict(a, b, on=self.on)
            self.assertEqual(restored.maps, self.maps)
            self.assertEqual(actual, expected)
            self.assertEqual(restored.elo_config, self.an.fx.elo.cfg)
            for name, value in restored._training_options.items():
                self.assertEqual(value, getattr(self.an.match_model, name))
            for name in ("match", "map"):
                model = getattr(restored, f"{name}_model")
                original = getattr(self.an, f"{name}_model")
                X = getattr(self.an, f"{name}_ds").X
                np.testing.assert_array_equal(model.predict_proba(X), original.predict_proba(X))
            self.assertEqual(restored.resolve_team("MOUZ"), self.an.resolve_team("MOUZ"))

    def test_historical_prediction_retains_original_training_options(self):
        on = self.an.match_ds.dates[-60]
        a, b = self.an.known_teams()[:2]
        expected = self.an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            self.an.save(path)
            restored = Analyzer.load(path)
            self.assertEqual(restored.predict(a, b, on=on), expected)
            restored.save(path)
            self.assertEqual(Analyzer.load(path).predict(a, b, on=on), expected)

    def test_corrupt_history_is_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            self.an.save(path)
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            files["history.json"] += b" "
            with ZipFile(path, "w") as archive:
                for name, blob in files.items():
                    archive.writestr(name, blob)
            with patch("cs2edge.model.MatchPredictor.load", side_effect=AssertionError("trop tot")):
                with self.assertRaisesRegex(ValueError, "somme de controle.*history"):
                    Analyzer.load(path)

    def test_incompatible_schema_and_unknown_archive_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            for mode in ("schema", "version", "path"):
                self.an.save(path)
                with ZipFile(path) as archive:
                    files = {name: archive.read(name) for name in archive.namelist()}
                manifest = json.loads(files["manifest.json"])
                if mode == "schema":
                    manifest["schemas"]["match"].reverse()
                elif mode == "version":
                    manifest["version"] = 99
                else:
                    files["../unexpected"] = b"unexpected"
                files["manifest.json"] = json.dumps(manifest).encode()
                with ZipFile(path, "w") as archive:
                    for name, blob in files.items():
                        archive.writestr(name, blob)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    Analyzer.load(path)
            self.assertFalse((path.parent.parent / "unexpected").exists())

    def test_failed_save_preserves_previous_artifact_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2edge"
            self.an.save(path)
            previous = path.read_bytes()
            with patch("cs2edge.snapshot.os.replace", side_effect=OSError("disque indisponible")):
                with self.assertRaises(OSError):
                    self.an.save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
