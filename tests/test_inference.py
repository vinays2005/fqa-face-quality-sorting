import importlib.util, pytest


MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
CV2 = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not (MEDIAPIPE and CV2), reason="mediapipe/cv2 not installed")
def test_inference_module_imports():
    from fqa.inference import extract, write_jsonl, CrFiqaAdapter
    assert callable(extract) and callable(write_jsonl)
    assert CrFiqaAdapter(None, None).available is False