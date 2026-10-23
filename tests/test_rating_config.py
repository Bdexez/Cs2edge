import unittest

from cs2edge.ratings import EloConfig


class RatingConfigTests(unittest.TestCase):
    def test_invalid_configurations_fail_before_rating_updates(self):
        for values in (dict(idle_halflife_days=0), dict(rd_growth_days=-1),
                       dict(rd_reference=0), dict(rd_min=301), dict(rd_initial=float("nan")),
                       dict(k_base=-1), dict(map_prior_weight=-.1), dict(roster_shock=1.1),
                       dict(lan_bonus=float("inf")), dict(use_mov="false"), dict(fast_multiplier=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EloConfig.from_dict(values)

    def test_disabled_updates_and_fast_horizon_are_valid(self):
        cfg = EloConfig(k_base=0, k_map=0, fast_multiplier=0, map_prior_weight=0,
                        use_uncertainty=False, lan_bonus=-1)
        self.assertEqual(EloConfig.from_dict(cfg.to_dict()), cfg)


if __name__ == "__main__":
    unittest.main()
