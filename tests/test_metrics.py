import pytest

from sensation_seminar.metrics.confusion import confusion_matrix
from sensation_seminar.metrics.safety import safety_metrics
from sensation_seminar.metrics.segmentation import metrics_from_confusion


def test_ignore_and_road_sidewalk_metrics() -> None:
    target = [1, 1, 1, 2, 2, 255]
    prediction = [1, 2, 2, 2, 1, 0]
    matrix = confusion_matrix(target, prediction)
    assert sum(map(sum, matrix)) == 5
    metrics = metrics_from_confusion(matrix)
    safety = safety_metrics(matrix)
    assert metrics["road_iou"] == pytest.approx(0.25)
    assert metrics["sidewalk_iou"] == pytest.approx(0.25)
    assert safety["road_as_sidewalk_error_rate"] == pytest.approx(2 / 3)
    assert safety["sidewalk_as_road_error_rate"] == pytest.approx(1 / 2)


def test_confusion_rejects_invalid_labels_and_shapes() -> None:
    with pytest.raises(ValueError):
        confusion_matrix([9], [0])
    with pytest.raises(ValueError):
        confusion_matrix([1, 2], [1])
    with pytest.raises(ValueError):
        metrics_from_confusion([[1]])
