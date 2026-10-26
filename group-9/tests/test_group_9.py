from group_9.result_analysis import analyze_results


def _result(experiment, seed, miou):
    return {"schema_version": "1.0", "run_id": f"{experiment}-{seed}", "status": "completed", "experiment": experiment, "seed": seed, "split": "validation", "git": {}, "environment": {}, "data": {}, "training": {}, "metrics": {"confusion_matrix": [[0] * 9 for _ in range(9)], "miou": miou, "road_iou": miou, "sidewalk_iou": miou, "road_as_sidewalk_error_rate": 0}}


def test_analysis_accepts_any_available_result_subset():
    rows = analyze_results([_result("random_budget_0.25x", 1, .4), _result("method", 1, .6)])
    method = next(row for row in rows if row["experiment"] == "method")
    assert abs(method["miou_delta_vs_baseline"] - .2) < 1e-12
