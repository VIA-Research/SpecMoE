import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_greedy import check_batches, compare_outputs


class GreedyCheckTests(unittest.TestCase):
    def test_identical_sequences_pass(self):
        self.assertEqual(compare_outputs([[4, 2], [5, 6, 2]], [[4, 2], [5, 6, 2]], 2), [])

    def test_reports_token_and_length_differences(self):
        differences = compare_outputs([[4, 2], [5, 6, 2]], [[7, 2], [5, 6]], 2)
        self.assertEqual([(item["sequence"], item["token_index"]) for item in differences],
                         [(1, 0), (2, 2)])

    def test_missing_or_extra_sequences_cannot_pass(self):
        for expected, actual in (([[4, 2]], []), ([], [[4, 2]]),
                                 ([[4, 2]], [[4, 2], [5, 2]]), ([], [])):
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaisesRegex(ValueError, "Output count differs"):
                    compare_outputs(expected, actual, 1)

    def test_incomplete_dataset_selection_fails_before_model_loading(self):
        with patch("scripts.check_greedy._batches", return_value=[["a", "b"]]), \
                patch("scripts.check_greedy.SpecDec") as model:
            with self.assertRaisesRegex(ValueError, "Requested 3 samples, but selected 2"):
                check_batches(SimpleNamespace(num_samples=3), {})
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
