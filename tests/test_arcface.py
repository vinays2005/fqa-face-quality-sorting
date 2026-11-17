import numpy as np
import pytest

from fqa.arcface import align_face, preprocess_arcface


def test_arcface_preprocess_nchw_shape_and_range():
    image = np.full((112, 112, 3), 255, dtype=np.uint8)
    tensor = preprocess_arcface(image, "NCHW")
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor, 1.0)


def test_arcface_preprocess_nhwc_shape():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    tensor = preprocess_arcface(image, "NHWC")
    assert tensor.shape == (1, 112, 112, 3)
    assert np.allclose(tensor, -1.0)


def test_align_face_bbox_fallback():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    result = align_face(image, [20, 30, 120, 130])
    assert result.shape == (112, 112, 3)


def test_align_face_rejects_bad_landmarks():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        align_face(image, [20, 30, 120, 130], [[1, 2], [3, 4]])
