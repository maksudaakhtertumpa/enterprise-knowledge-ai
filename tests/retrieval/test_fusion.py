import pytest

from app.services.retrieval.fusion import reciprocal_rank_fusion as rrf


def ids(fused):
    return [f.id for f in fused]


def test_hand_computed_example():
    fused = rrf({"bm25": ["a", "b", "c"], "dense": ["c", "a", "d"]}, k=60)
    expected = {
        "a": 1 / 61 + 1 / 62,
        "c": 1 / 63 + 1 / 61,
        "b": 1 / 62,
        "d": 1 / 63,
    }
    assert ids(fused) == ["a", "c", "b", "d"]
    for item in fused:
        assert item.score == pytest.approx(expected[item.id])
    assert fused[0].ranks == {"bm25": 1, "dense": 2}
    assert fused[2].ranks == {"bm25": 2}


def test_single_ranking_keeps_its_order():
    assert ids(rrf({"bm25": ["x", "y", "z"]})) == ["x", "y", "z"]


def test_agreement_beats_a_single_first_place():
    # "both" is 2nd in each list; "solo" is 1st in only one.
    fused = rrf({"bm25": ["solo", "both"], "dense": ["other", "both"]})
    assert fused[0].id == "both"


def test_k_controls_how_much_the_top_rank_dominates():
    rankings = {"bm25": ["top", "f1", "f2", "shared"], "dense": ["g1", "g2", "g3", "shared"]}
    # Small k: a lone 1st place (1/2) beats 4th place in both lists (2/5).
    assert rrf(rankings, k=1)[0].id in {"top", "g1"}
    # Paper default k=60: consensus wins (2/64 > 1/61).
    assert rrf(rankings, k=60)[0].id == "shared"


def test_weights():
    rankings = {"bm25": ["a", "b"], "dense": ["b", "a"]}
    assert ids(rrf(rankings, weights={"bm25": 2.0, "dense": 1.0}))[0] == "a"
    assert ids(rrf(rankings, weights={"bm25": 1.0, "dense": 2.0}))[0] == "b"
    # Zero weight: that retriever cannot change the order.
    only_bm25 = rrf({"bm25": ["a", "b", "c"], "dense": ["c", "b"]}, weights={"dense": 0.0})
    assert ids(only_bm25) == ["a", "b", "c"]


def test_ties_are_broken_deterministically():
    fused = rrf({"bm25": ["z"], "dense": ["m"]})
    assert fused[0].score == fused[1].score
    assert ids(fused) == ["m", "z"]  # equal score and best rank -> by id
    assert ids(rrf({"dense": ["m"], "bm25": ["z"]})) == ["m", "z"]  # input order irrelevant


def test_duplicates_within_one_list_count_once():
    fused = rrf({"bm25": ["a", "a", "b"]})
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].ranks == {"bm25": 3}


def test_empty_inputs():
    assert rrf({}) == []
    assert rrf({"bm25": [], "dense": []}) == []
    assert ids(rrf({"bm25": [], "dense": ["a"]})) == ["a"]


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -5}, {"weights": {"bm25": -1.0}}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rrf({"bm25": ["a"]}, **kwargs)
