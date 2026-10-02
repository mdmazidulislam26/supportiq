# Tests for the recall-first vs balanced threshold calibration.
from factory import calibrate_threshold


# Recall-first threshold lets every dev answerable question through.
def test_recall_first_keeps_every_dev_answerable(stack):
    retriever, _ = stack
    info = calibrate_threshold(retriever, save=False, strategy="recall_first")
    assert info["answerable_recall_dev"] == 1.0
    assert info["threshold"] <= info["answerable_score_min"]


# The balanced strategy is still selectable for before/after comparison.
def test_balanced_strategy_still_available(stack):
    retriever, _ = stack
    info = calibrate_threshold(retriever, save=False, strategy="balanced")
    assert info["strategy"] == "balanced" and info["threshold"] > 0
