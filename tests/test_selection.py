import pytest

from sensation_seminar.synthetic_data.selection import budget_size, random_training_subset


def test_canonical_budgets_and_deterministic_eligibility_filter() -> None:
    assert budget_size("0.25x") == 548
    assert budget_size("0.50x") == 1095
    records = [
        {
            "synthetic_id": f"id-{index:04d}",
            "source_group": f"group{1 + index % 2}",
            "eligibility": {"segmentation_training": index < 600},
        }
        for index in range(700)
    ]
    first = random_training_subset(records, budget="0.25x", seed=42)
    second = random_training_subset(reversed(records), budget="0.25x", seed=42)
    assert first == second
    assert len(first) == 548
    assert {row["budget"] for row in first} == {"budget_0.25x"}
    assert all(int(row["synthetic_id"].split("-")[1]) < 600 for row in first)


def test_insufficient_eligible_records_fail_clearly() -> None:
    with pytest.raises(ValueError, match="training-eligible"):
        random_training_subset([], budget="0.25x")
