from group_1.audit import audit


def test_audit_counts_eligibility_and_supervision():
    rows = [{"source_group": "group1", "validity": {"reason_codes": []}, "eligibility": {"analysis": True, "segmentation_training": True}, "supervision": {"status": "verified"}}, {"source_group": "group2", "validity": {"reason_codes": ["artifact"]}, "eligibility": {"analysis": False, "segmentation_training": False}, "supervision": {"status": "invalid"}}]
    result = audit(rows)
    assert result["records"] == 2 and result["training_eligible"] == 1
    assert result["supervision"] == {"invalid": 1, "verified": 1}
