# Installation

## Requirements

- Windows 10/11
- Python 3.10–3.12 (use a version supported by the installed MediaPipe release)
- Optional NVIDIA GPU for CR-FIQA acceleration; the base FQA pipeline can run without PyTorch

## Create the environment

```powershell
cd path\to\FQA_Face_Quality_Sorting_System
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
in that terminal, then activate again.

## Download the MediaPipe model

The ZIP does not contain pretrained model weights. Create the models directory and download:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile "models/face_landmarker.task"
```

## Optional CR-FIQA

CR-FIQA weights are not distributed with this package. Obtain a compatible TorchScript
model from its upstream source. Install a PyTorch wheel that matches your CUDA setup
using the official PyTorch selector, then install `requirements-crfiqa.txt`.

Save the model as `models/crfiqa/crfiqa.pt` and calibration quantiles as
`models/crfiqa/calibration.json`. See `docs/MODELS.md`.

## Existing detector and ArcFace pipeline

This project does not include or run SCRFD or ArcFace. It consumes the bounding boxes,
confidence values, and cluster IDs already produced by your pipeline. See
`docs/INTEGRATION.md` before processing a real dataset.
