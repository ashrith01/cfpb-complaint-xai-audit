"""
Tests for the faithfulness scoring primitives.

These are the functions the project's headline finding rests on. A sign error in
top_k_indices or a mistake in _rebuild would not crash anything -- it would
quietly reorder the method ranking and produce a confident, wrong conclusion.
"""

import pytest

from src.audit import merge_wordpieces, top_k_overlap, top_k_words
from src.faithfulness import TOP_K_PCT, _rebuild, top_k_indices


class TestTopKSelection:
    def test_selects_highest_signed_scores(self):
        assert set(top_k_indices([0.1, 0.9, 0.2, 0.8], k_pct=0.5)) == {1, 3}

    def test_ranks_by_signed_not_absolute(self):
        """A large negative score argues AGAINST the prediction; removing it
        should raise confidence, so it must not be selected as top evidence."""
        assert set(top_k_indices([-5.0, 0.4, 0.1], k_pct=0.34)) == {1}

    def test_always_returns_at_least_one(self):
        assert len(top_k_indices([0.5, 0.2], k_pct=0.001)) == 1

    def test_default_fraction(self):
        assert len(top_k_indices(list(range(100)))) == int(round(100 * TOP_K_PCT))


class TestRebuild:
    def test_keeps_only_requested_indices(self):
        assert _rebuild(["a", "b", "c"], {0, 2}, None) == "a c"

    def test_rejoins_wordpiece_continuations(self):
        assert _rebuild(["walk", "##ing"], {0, 1}, None) == "walking"

    def test_dropping_a_word_leaves_no_orphan_fragment(self):
        out = _rebuild(["the", "walk", "##ing", "dog"], {0, 3}, None)
        assert out == "the dog"

    def test_empty_selection(self):
        assert _rebuild(["a", "b"], set(), None) == ""


class TestWordSpaceComparison:
    def test_wordpieces_merged_and_summed(self):
        merged = merge_wordpieces(["walk", "##ing", "dog"], [0.3, 0.2, 0.9])
        assert merged == pytest.approx({"walking": 0.5, "dog": 0.9})

    def test_repeated_word_keeps_strongest(self):
        merged = merge_wordpieces(["loan", "loan"], [0.2, 0.7])
        assert merged["loan"] == pytest.approx(0.7)

    def test_case_normalised(self):
        assert "zelle" in merge_wordpieces(["Zelle"], [1.0])

    def test_overlap_identical_is_one(self):
        a = {"x": 3.0, "y": 2.0, "z": 1.0}
        assert top_k_overlap(a, dict(a), k=3) == 1.0

    def test_overlap_disjoint_is_zero(self):
        a = {"x": 3.0, "y": 2.0, "z": 1.0}
        b = {"p": 3.0, "q": 2.0, "r": 1.0}
        assert top_k_overlap(a, b, k=3) == 0.0

    def test_overlap_partial(self):
        a = {"x": 3.0, "y": 2.0, "z": 1.0}
        b = {"x": 3.0, "y": 2.0, "w": 1.0}
        assert top_k_overlap(a, b, k=3) == pytest.approx(2 / 3)

    def test_overlap_handles_empty(self):
        assert top_k_overlap({}, {"a": 1.0}) == 0.0

    def test_top_k_words_picks_highest(self):
        assert top_k_words({"a": 1.0, "b": 5.0, "c": 3.0}, k=2) == {"b", "c"}
