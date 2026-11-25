"""Per-frame colour measurement: skin (forehead + both cheeks) and a background ring around the face."""
import cv2
import numpy as np

# YuNet output row: x, y, w, h, right eye (x, y), left eye, nose tip, right mouth corner, left mouth corner, score
_EYES = slice(4, 8)


class FaceRegions:
    def __init__(self, model_path, score_threshold):
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)

    def measure(self, frame_bgr):
        """(skin_rgb, background_rgb) mean colours, or (None, None) if no face is found."""
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None, None
        face = faces[np.argmax(faces[:, -1])]
        rgb = frame_bgr[:, :, ::-1]
        skin = [_mean(rgb, box) for box in skin_boxes(face, w, h)]
        skin = [m for m in skin if m is not None]
        return (np.mean(skin, axis=0) if skin else None), _ring_mean(rgb, face[:4], w, h)


def skin_boxes(face, width, height):
    """Forehead and cheek boxes (x0, y0, x1, y1), placed from the eye landmarks and sized by eye distance."""
    (rex, rey, lex, ley) = face[_EYES]
    d = max(abs(lex - rex), 1.0)  # eye distance: scales every box with the face
    eye_y = (rey + ley) / 2
    left_x, right_x = min(rex, lex), max(rex, lex)
    boxes = [
        (left_x, eye_y - 0.65 * d, right_x, eye_y - 0.3 * d),                      # forehead, above the brows
        (left_x - 0.25 * d, eye_y + 0.35 * d, left_x + 0.2 * d, eye_y + 0.75 * d),   # cheek under one eye
        (right_x - 0.2 * d, eye_y + 0.35 * d, right_x + 0.25 * d, eye_y + 0.75 * d), # cheek under the other
    ]
    return [_clip(b, width, height) for b in boxes]


def _ring_mean(rgb, bbox, width, height, inner=1.3, outer=1.8):
    """Mean colour between the face box scaled by `inner` and by `outer` (hair and shoulders mostly excluded)."""
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0, x1, y1 = _clip((cx - bw * outer / 2, cy - bh * outer / 2, cx + bw * outer / 2, cy + bh * outer / 2),
                           width, height)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.ones((y1 - y0, x1 - x0), bool)
    ix0, iy0, ix1, iy1 = _clip((cx - bw * inner / 2, cy - bh * inner / 2, cx + bw * inner / 2, cy + bh * inner / 2),
                               width, height)
    mask[max(iy0 - y0, 0):max(iy1 - y0, 0), max(ix0 - x0, 0):max(ix1 - x0, 0)] = False
    pixels = rgb[y0:y1, x0:x1][mask]
    return pixels.mean(axis=0) if len(pixels) else None


def _clip(box, width, height):
    x0, y0, x1, y1 = box
    return (int(np.clip(x0, 0, width)), int(np.clip(y0, 0, height)),
            int(np.clip(x1, 0, width)), int(np.clip(y1, 0, height)))


def _mean(rgb, box):
    x0, y0, x1, y1 = box
    patch = rgb[y0:y1, x0:x1]
    return patch.reshape(-1, 3).mean(axis=0) if patch.size else None
