import datetime as dt
import unittest

import numpy as np

from cs2edge.metrics import log_loss, paired_block_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.y = np.tile([1, 0, 0, 1], 10)
        self.p = np.linspace(.2, .8, 40)
        self.reference = np.full(40, .5)
        self.dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]

    def test_gain_is_actual_paired_log_loss_difference_and_reproducible(self):
        result = paired_block_bootstrap(self.y, self.p, self.reference, self.dates)
        self.assertAlmostEqual(result["gain"], log_loss(self.y, self.reference)-log_loss(self.y, self.p))
        self.assertEqual(result, paired_block_bootstrap(self.y, self.p, self.reference, self.dates))
        same = paired_block_bootstrap(self.y, self.p, self.p, self.dates)
        self.assertEqual((same["gain"], same["low"], same["high"]), (0, 0, 0))

    def test_duplicating_correlated_rows_does_not_narrow_the_interval(self):
        before = paired_block_bootstrap(self.y, self.p, self.reference, self.dates)
        after = paired_block_bootstrap(np.repeat(self.y, 3), np.repeat(self.p, 3),
                                      np.repeat(self.reference, 3),
                                      [d for d in self.dates for _ in range(3)])
        for key in ("gain", "low", "high"):
            self.assertAlmostEqual(before[key], after[key], places=12)
        self.assertLess(before["low"], before["high"])

    def test_no_interval_is_invented_for_one_block(self):
        result = paired_block_bootstrap([1, 0], [.8, .2], [.5, .5], [self.dates[0]] * 2)
        self.assertGreater(result["gain"], 0)
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_invalid_resampling_inputs_are_rejected(self):
        for options in (dict(samples=0), dict(block_days=0), dict(samples=1.5)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired_block_bootstrap(self.y, self.p, self.reference, self.dates, **options)
        with self.assertRaises(ValueError):
            paired_block_bootstrap(self.y, self.p, self.reference, self.dates[:-1])


if __name__ == "__main__":
    unittest.main()
