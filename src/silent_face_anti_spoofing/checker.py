"""LivenessChecker: the one class a caller needs. detect face -> crop per model -> average the models' votes."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from silent_face_anti_spoofing.inference.cropper import CropImage
from silent_face_anti_spoofing.inference.detector import FaceDetector
from silent_face_anti_spoofing.inference.predictor import AntiSpoofModel

# shortcut: resolves to <repo>/resources, so it only works from a source checkout / editable install;
# pass model_dir and detector_dir explicitly (or ship weights as package data) when deploying a wheel.
RESOURCES = Path(__file__).resolve().parents[2] / "resources"
REAL_LABEL = 1


@dataclass
class LivenessResult:
    is_real: bool
    score: float      # averaged probability of the winning class
    label: int        # 1 = real, 0 / 2 = spoof types
    bbox: list        # [x, y, w, h] of the face that was checked


class LivenessChecker:
    def __init__(self, model_dir=RESOURCES / "anti_spoof_models",
                 detector_dir=RESOURCES / "detection_model", device_id=0):
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.detector = FaceDetector(detector_dir)
        self.cropper = CropImage()
        self.models = [AntiSpoofModel(p, device) for p in sorted(Path(model_dir).glob("*.pth"))]
        if not self.models:
            raise FileNotFoundError(f"no .pth model files in {model_dir}")

    def check(self, image):
        """image: BGR uint8 array (as from cv2.imread). Returns LivenessResult, or None if no face is found."""
        bbox = self.detector.get_bbox(image)
        if bbox is None:
            return None
        probs = np.mean([
            m.predict(self.cropper.crop(image, bbox, m.scale, m.width, m.height, crop=m.scale is not None))
            for m in self.models
        ], axis=0)
        label = int(np.argmax(probs))
        return LivenessResult(is_real=label == REAL_LABEL, score=float(probs[label]), label=label, bbox=bbox)
