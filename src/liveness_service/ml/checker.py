"""LivenessChecker: the one class a caller needs. detect face -> crop per model -> average the models' votes."""
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from liveness_service.ml.inference.cropper import CropImage
from liveness_service.ml.inference.detector import FaceDetector
from liveness_service.ml.inference.predictor import AntiSpoofModel
from liveness_service.exceptions import VideoUnreadableError

# shortcut: resolves to <repo>/resources, so it only works from a source checkout / editable install;
# pass model_dir and detector_dir explicitly (or ship weights as package data) when deploying a wheel.
RESOURCES = Path(__file__).resolve().parents[3] / "resources"
REAL_LABEL = 1


@dataclass
class LivenessResult:
    is_real: bool
    score: float      # averaged probability of the winning class
    label: int        # 1 = real, 0 / 2 = spoof types
    bbox: list        # [x, y, w, h] of the face that was checked


@dataclass
class VideoLivenessResult:
    is_real: bool          # more than half of the frames with a face were judged real
    real_ratio: float      # share of frames with a face that were judged real
    frames_with_face: int
    frames_sampled: int


class LivenessChecker:
    def __init__(self, model_dir=RESOURCES / "anti_spoof_models",
                 detector_dir=RESOURCES / "detection_model", device_id=0):
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.detector = FaceDetector(detector_dir)
        self.cropper = CropImage()
        self.models = [AntiSpoofModel(p, device) for p in sorted(Path(model_dir).glob("*.pth"))]
        if not self.models:
            raise FileNotFoundError(f"no .pth model files in {model_dir}")
        self._lock = threading.Lock()  # the OpenCV face detector is not thread-safe

    def check(self, image):
        """image: BGR uint8 array (as from cv2.imread). Returns LivenessResult, or None if no face is found."""
        with self._lock:
            bbox = self.detector.get_bbox(image)
            if bbox is None:
                return None
            probs = np.mean([
                m.predict(self.cropper.crop(image, bbox, m.scale, m.width, m.height, crop=m.scale is not None))
                for m in self.models
            ], axis=0)
        label = int(np.argmax(probs))
        return LivenessResult(is_real=label == REAL_LABEL, score=float(probs[label]), label=label, bbox=bbox)

    def check_video(self, path, max_frames=10):
        """Check up to max_frames frames spread across the video. Returns VideoLivenessResult, or None if no
        sampled frame has a face. Raises VideoUnreadableError if the file can't be opened as a video."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoUnreadableError("could not open video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames) if total > 0 else 5  # some containers (e.g. webm) report no count
        results, sampled, index = [], 0, 0
        try:
            while sampled < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    sampled += 1
                    results.append(self.check(frame))
                index += 1
        finally:
            cap.release()
        faces = [r for r in results if r is not None]
        if not faces:
            return None
        real = sum(r.is_real for r in faces)
        # strict majority: a tie counts as fake
        return VideoLivenessResult(is_real=real * 2 > len(faces), real_ratio=real / len(faces),
                                   frames_with_face=len(faces), frames_sampled=sampled)
