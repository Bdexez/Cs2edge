import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from cs2edge.cli import build_parser, main


class CommonOptionsTests(unittest.TestCase):
    def test_options_work_on_either_side_of_subcommand(self):
        parser = build_parser()
        for option, value, attribute in (
            ("--data", "custom.csv", "data"),
            ("--warmup", "25", "warmup"),
            ("--elo-config", "custom.json", "elo_config"),
        ):
            with self.subTest(option=option):
                before = parser.parse_args([option, value, "teams"])
                after = parser.parse_args(["teams", option, value])
                self.assertEqual(getattr(before, attribute), getattr(after, attribute))
        self.assertTrue(parser.parse_args(["--no-gbdt", "teams"]).no_gbdt)
        self.assertTrue(parser.parse_args(["teams", "--no-gbdt"]).no_gbdt)
        args = parser.parse_args(["--warmup", "25", "teams", "--warmup", "30"])
        self.assertEqual(args.warmup, 30)

    def test_actual_command_reads_requested_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.csv"
            path.write_text("match_id,date,team_a,team_b,map,rounds_a,rounds_b,best_of\n"
                            "one,2026-01-01,alpha,beta,mirage,13,5,1\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--data", str(path), "teams"]), 0)
            self.assertIn("alpha", out.getvalue())
            self.assertNotIn("navi", out.getvalue())

    def test_empty_history_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("match_id,date,team_a,team_b,map,rounds_a,rounds_b\n")
            with self.assertRaisesRegex(SystemExit, "aucun resultat"):
                main(["teams", "--data", str(path)])


if __name__ == "__main__":
    unittest.main()
