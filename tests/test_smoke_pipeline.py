"""CPU-only smoke path using the permitted lightweight model equivalent."""

from sensation_seminar.metrics.confusion import confusion_matrix
from sensation_seminar.metrics.safety import safety_metrics
from sensation_seminar.metrics.segmentation import metrics_from_confusion
from sensation_seminar.results.io import write_result_bundle


class TinyMockSegmenter:
    def __init__(self) -> None:
        self.steps = 0

    def train_batch(self, _images, _masks) -> float:
        self.steps += 1
        return 1.0 / self.steps

    def predict(self, images):
        return [[pixel % 9 for pixel in image] for image in images]


def test_two_batch_training_evaluation_and_result_bundle(tmp_path) -> None:
    model = TinyMockSegmenter()
    assert model.train_batch([[0, 1]], [[0, 1]]) == 1.0
    assert model.train_batch([[1, 2]], [[1, 2]]) == 0.5
    targets = [[0, 1, 2, 255]]
    predictions = model.predict([[0, 1, 2, 3]])
    matrix = confusion_matrix(targets, predictions)
    metrics = metrics_from_confusion(matrix)
    metrics.update(safety_metrics(matrix))
    result = {
        "schema_version": "1.0", "run_id": "smoke", "status": "completed",
        "experiment": "mock_smoke", "seed": 17, "split": "testing",
        "git": {"revision": "fixture", "dirty": False},
        "environment": {"python": "fixture"}, "data": {"dataset": "tiny"},
        "training": {"checkpoint_selected_on_validation": True, "batches": 2},
        "metrics": metrics,
    }
    output = write_result_bundle(tmp_path / "result", result, {"model": "mock"})
    assert (output / "metrics.json").is_file()
    assert (output / "checkpoint_reference.json").is_file()
