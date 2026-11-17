# Model weights (NOT bundled in the ZIP)

MediaPipe Face Landmarker (required for visibility / pose)
Download face_landmarker.task:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Save to models/face_landmarker.task. Verify with:

powershell
python -c "import mediapipe, pathlib; print(pathlib.Path('models/face_landmarker.task').stat().st_size)"
Expected size: ~3.7 MB.

CR-FIQA (optional)
Not redistributed. Obtain from the upstream author's release, export to
TorchScript (torch.jit.trace), save as models/crfiqa/crfiqa.pt.

Install optional PyTorch dependencies from requirements-crfiqa.txt before using this adapter.\n\nCalibration (required before use)
python
import cv2, json, numpy as np, torch
model = torch.jit.load("models/crfiqa/crfiqa.pt").eval()
scores = []
for path in calibration_images:  # ~200 real crops from your dataset
    img = cv2.resize(cv2.imread(path), (112, 112))
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) \\
             .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        scores.append(float(model(t).mean()))
lo, hi = np.percentile(scores, [5, 95])
json.dump({"p05": float(lo), "p95": float(hi)},
          open("models/crfiqa/calibration.json", "w"))
If calibration is missing, crfiqa_status becomes "uncalibrated" and the
weight is ignored — the engine refuses to invent a number.

SCRFD / ArcFace
Reused from your existing pipeline. This package does not download or load
them.