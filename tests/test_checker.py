from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from silent_face_anti_spoofing import LivenessChecker
from silent_face_anti_spoofing.inference.predictor import AntiSpoofModel
from silent_face_anti_spoofing.training.multiftnet import MultiFTNet

SAMPLES = Path(__file__).resolve().parents[1] / "images" / "sample"


@pytest.fixture(scope="module")
def checker():
    return LivenessChecker()


# expected scores recorded from the original test.py pipeline (OpenCV 4.11); newer OpenCV shifts the
# face box by ~1px, moving scores by up to ~0.01, so the tolerance allows that but catches real breakage
@pytest.mark.parametrize("name, is_real, score", [
    ("image_F1.jpg", False, 0.732),
    ("image_F2.jpg", False, 0.817),
    ("image_T1.jpg", True, 0.994),
])
def test_sample_images(checker, name, is_real, score):
    result = checker.check(cv2.imread(str(SAMPLES / name)))
    assert result.is_real == is_real
    assert result.score == pytest.approx(score, abs=0.02)


def test_no_face_returns_none(checker):
    assert checker.check(np.full((640, 480, 3), 128, np.uint8)) is None


def test_loads_checkpoint_saved_by_training(tmp_path):
    # training saves DataParallel(MultiFTNet).state_dict() under a name parse_model_name understands
    path = tmp_path / "1_2026-11-24-10-30-iter-30_80x80_MiniFASNetV2SE.pth"
    torch.save(torch.nn.DataParallel(MultiFTNet(conv6_kernel=(5, 5))).state_dict(), path)
    probs = AntiSpoofModel(path, torch.device("cpu")).predict(np.zeros((80, 80, 3), np.uint8))
    assert probs.shape == (3,)
