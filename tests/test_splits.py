import unittest
from datetime import timedelta

import pandas as pd

from jeena_sikho_tournament.splits import walk_forward_split


class TestSplits(unittest.TestCase):
    def test_walk_forward(self):
        idx = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz="UTC")
        df = pd.DataFrame({"x": range(len(idx))}, index=idx)
        split = walk_forward_split(df, train_days=90, val_hours=72, test_hours=24, use_test=True)
        self.assertTrue(len(split.train) > 0)
        self.assertTrue(len(split.val) > 0)
        self.assertTrue(len(split.test) > 0)
        self.assertTrue(split.train.index.max() < split.val.index.min())
        self.assertTrue(split.val.index.max() <= split.test.index.max())


if __name__ == "__main__":
    unittest.main()

