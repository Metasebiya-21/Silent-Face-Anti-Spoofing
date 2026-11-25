"""Face detection: find the most confident face box in a BGR image (RetinaFace, Caffe, via OpenCV DNN)."""
import math
from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, detector_dir, confidence=0.6):
        detector_dir = Path(detector_dir)
        self.net = cv2.dnn.readNetFromCaffe(str(detector_dir / "deploy.prototxt"),
                                            str(detector_dir / "Widerface-RetinaFace.caffemodel"))
        self.confidence = confidence

    def get_bbox(self, img):
        """Return [x, y, w, h] of the most confident face, or None if no face reaches the threshold."""
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if width * height >= 192 * 192:  # detector runs on ~192x192 for speed; outputs are 0-1 fractions
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.net.setInput(blob, 'data')
        out = self.net.forward('detection_out').reshape(-1, 7)  # squeeze() breaks on a single detection
        if len(out) == 0:
            return None
        best = out[np.argmax(out[:, 2])]
        if best[2] < self.confidence:
            return None
        left, top, right, bottom = best[3] * width, best[4] * height, best[5] * width, best[6] * height
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
