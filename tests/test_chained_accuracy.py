"""
Tests for the chained accuracy metric.

These reproduce the five worked examples (A, B, C, D, E) from the
assignment brief so we can be certain the metric is computed exactly as
specified.
"""
import unittest

from src.metrics import chained_accuracy, chained_accuracy_per_row


class BriefExampleTests(unittest.TestCase):
    """
    One row, three label levels.
    Ground truth = (Suggestion, Payment, Subscription Cancelled).
    """
    TRUE_CHAIN = [["Suggestion"], ["Payment"], ["Subscription Cancelled"]]

    def test_model_A_all_correct_is_100_percent(self):
        # all three labels match -> 1.0
        pred = [["Suggestion"], ["Payment"], ["Subscription Cancelled"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(self.TRUE_CHAIN, pred)[0], 1.0, places=6,
        )

    def test_model_B_last_wrong_is_67_percent(self):
        # first two right, last wrong -> 2/3
        pred = [["Suggestion"], ["Payment"], ["Subscription Retained"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(self.TRUE_CHAIN, pred)[0], 2 / 3, places=6,
        )

    def test_model_C_only_first_correct_is_33_percent(self):
        # only y2 right -> 1/3, even though we got further down the chain
        pred = [["Suggestion"], ["Refund"], ["Subscription Retained"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(self.TRUE_CHAIN, pred)[0], 1 / 3, places=6,
        )

    def test_model_D_first_wrong_is_0_percent(self):
        # y2 wrong -> whole chain scores zero even though y3 and y4 "match"
        pred = [["Other"], ["Payment"], ["Subscription Cancelled"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(self.TRUE_CHAIN, pred)[0], 0.0, places=6,
        )

    def test_model_E_y2_right_y3_wrong_y4_matches_is_33_percent(self):
        # Brief's tricky case: y4 matches the ground truth but doesn't count
        # because the chain already broke at y3
        pred = [["Suggestion"], ["Refund"], ["Subscription Cancelled"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(self.TRUE_CHAIN, pred)[0], 1 / 3, places=6,
        )


class AggregationTests(unittest.TestCase):
    def test_mean_over_multiple_rows(self):
        true_chain = [
            ["Suggestion", "Suggestion", "Suggestion"],
            ["Payment", "Payment", "Payment"],
            ["Sub Cancelled", "Sub Cancelled", "Sub Cancelled"],
        ]
        # row 0: all right (1.0), row 1: y2 wrong (0.0), row 2: y3 wrong (1/3)
        pred_chain = [
            ["Suggestion", "Other", "Suggestion"],
            ["Payment", "Payment", "Refund"],
            ["Sub Cancelled", "Sub Cancelled", "Sub Cancelled"],
        ]
        expected = (1.0 + 0.0 + 1 / 3) / 3
        self.assertAlmostEqual(
            chained_accuracy(true_chain, pred_chain), expected, places=6,
        )

    def test_missing_label_shortens_chain(self):
        # row has no ground truth at y4, so chain is effectively length 2
        true_chain = [["Suggestion"], ["Payment"], ["__MISSING__"]]
        pred_chain = [["Suggestion"], ["Payment"], ["Anything"]]
        self.assertAlmostEqual(
            chained_accuracy_per_row(true_chain, pred_chain)[0], 1.0, places=6,
        )

if __name__ == "__main__":
    unittest.main()
