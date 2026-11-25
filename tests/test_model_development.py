import numpy as np
import pytest
from PIL import Image

from model_development.transforms import functional as F
from model_development.transforms import transform as T


@pytest.mark.parametrize("hue", [-0.1, -0.01, 0.0, 0.01, 0.1])
def test_hue_shift_works_both_ways_on_numpy_2(hue):
    # np.uint8(negative) raises on NumPy 2; this used to break ~half of all training images
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 16, 3), np.uint8))
    assert np.asarray(F.adjust_hue(image, hue)).shape == (16, 16, 3)


def test_training_augmentation_keeps_0_255_pixels():
    augment = T.Compose([T.ToPILImage(), T.RandomResizedCrop(size=(80, 80), scale=(0.9, 1.1)),
                         T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomRotation(10), T.RandomHorizontalFlip(), T.ToTensor()])
    image = np.full((100, 90, 3), 200, np.uint8)
    for _ in range(20):  # random hue each time, negative included
        x = augment(image)
        assert tuple(x.shape) == (3, 80, 80) and x.max() > 1  # no /255: the model was trained on 0-255
