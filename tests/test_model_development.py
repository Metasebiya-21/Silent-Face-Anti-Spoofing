import numpy as np

from tests.notebooks import training

train_transform = training("data")["train_transform"]


def test_training_augmentation_keeps_0_255_pixels():
    augment = train_transform((80, 80))
    image = np.full((100, 90, 3), 200, np.uint8)  # BGR uint8, as cv2.imread gives
    for _ in range(20):  # random crop / jitter / rotation each time
        x = augment(image)
        assert tuple(x.shape) == (3, 80, 80) and x.dtype.is_floating_point
        assert x.max() > 1 and x.max() <= 255  # no /255: the model was trained and serves on 0-255
