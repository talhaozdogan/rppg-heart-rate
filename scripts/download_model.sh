#!/usr/bin/env bash
# Download the MediaPipe FaceLandmarker model bundle required by src/face_roi.py.
# (Not vendored in git — it is a ~3.6 MB binary asset.)
set -euo pipefail
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -sSL -o models/face_landmarker.task "$URL"
echo "Saved models/face_landmarker.task"
