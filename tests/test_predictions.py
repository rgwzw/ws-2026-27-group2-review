import numpy as np
import pytest
from PIL import Image

from sensation_seminar.evaluation.predictions import save_label_predictions


def test_prediction_export_is_lossless_and_rejects_invalid(tmp_path):
    labels = np.array([[0, 1], [8, 255]], dtype=np.uint8)
    paths = save_label_predictions([("sample", labels)], tmp_path)
    assert np.array_equal(np.asarray(Image.open(paths[0])), labels)
    with pytest.raises(ValueError):
        save_label_predictions([("bad", np.zeros((1, 1, 1)))], tmp_path)
