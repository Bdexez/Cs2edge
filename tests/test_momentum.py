import unittest

import numpy as np

from cs2edge.markets import _fit_momentum
from cs2edge.model import sigmoid


class MomentumEstimatorTests(unittest.TestCase):
    @staticmethod
    def samples(effect):
        rng = np.random.default_rng(81)
        samples = []
        for match in range(2500):
            balance, previous = 0, 0
            for index in range(5):
                z = rng.normal()
                p = float(sigmoid(z))
                won = int(rng.random() < sigmoid(z+effect*balance))
                if index:
                    samples.append((p, won, balance, previous, str(match)))
                balance += 1 if won else -1
                previous = won
        return samples

    def test_estimated_coefficient_matches_the_cumulative_effect_used_for_generation(self):
        for effect in (0., .35, -.35):
            with self.subTest(effect=effect):
                result = _fit_momentum(self.samples(effect))
                self.assertAlmostEqual(result["momentum"], effect, delta=.05)
                self.assertEqual(result["significatif"], effect != 0)
                self.assertGreater(result["erreur_type"], 0)

    def test_correlated_duplicates_do_not_narrow_the_standard_error(self):
        samples = self.samples(.35)
        before = _fit_momentum(samples)
        after = _fit_momentum([s for s in samples for _ in range(3)])
        self.assertAlmostEqual(before["momentum"], after["momentum"], places=12)
        self.assertAlmostEqual(before["erreur_type"], after["erreur_type"], places=12)

    def test_reversing_teams_preserves_effect_and_uncertainty(self):
        samples = self.samples(.35)
        before = _fit_momentum(samples)
        after = _fit_momentum([(1-p, 1-y, -balance, 1-previous, key)
                               for p, y, balance, previous, key in samples])
        self.assertAlmostEqual(before["momentum"], after["momentum"], places=12)
        self.assertAlmostEqual(before["erreur_type"], after["erreur_type"], places=12)


if __name__ == "__main__":
    unittest.main()
