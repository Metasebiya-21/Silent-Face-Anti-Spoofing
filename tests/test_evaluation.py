import math

import cv2
import numpy as np
import pytest
import torch

from tests.notebooks import training, val

evaluation = val("evaluation")
evaluate, evaluate_checkpoints, metrics = evaluation["evaluate"], evaluation["evaluate_checkpoints"], evaluation["metrics"]
MultiFTNet = training("model")["MultiFTNet"]


def test_metrics_counts_attacks_accepted_and_reals_rejected():
    # 4 attacks (labels 0/2), 4 reals (label 1)
    y_true = [0, 0, 2, 2, 1, 1, 1, 1]
    y_pred = [0, 1, 2, 2, 1, 1, 0, 1]  # one attack accepted, one real rejected
    m = metrics(y_true, y_pred)
    assert m["apcer"] == 0.25 and m["bpcer"] == 0.25 and m["acer"] == 0.25
    assert m["accuracy"] == 0.75 and m["samples"] == 8
    assert m["confusion_matrix"] == [[1, 1, 0], [1, 3, 0], [0, 0, 2]]


def test_metrics_without_attacks_or_reals_is_nan_not_a_crash():
    assert math.isnan(metrics([1, 1], [1, 0])["apcer"])
    assert math.isnan(metrics([0, 2], [0, 2])["bpcer"])


def test_evaluate_reads_a_split_folder_and_ranks_checkpoints(tmp_path):
    data = tmp_path / "val"
    for label in (0, 1, 2):
        (data / str(label)).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(data / str(label) / f"{i}.png"), np.full((80, 80, 3), 60 * (label + 1), np.uint8))
    (data / "1" / ".DS_Store").write_bytes(b"junk")  # skipped, not a crash
    snapshots = tmp_path / "snapshot"
    snapshots.mkdir()
    for step in (30, 60):
        torch.save(torch.nn.DataParallel(MultiFTNet(conv6_kernel=(5, 5))).state_dict(),
                   snapshots / f"2.7_2026-11-25-10-00-iter-{step}_80x80_MiniFASNetV2SE.pth")

    result = evaluate(next(snapshots.glob("*.pth")), data)
    assert result["samples"] == 6 and set(result) >= {"checkpoint", "apcer", "bpcer", "acer", "confusion_matrix"}
    ranked = evaluate_checkpoints(snapshots, data)
    assert len(ranked) == 2 and ranked[0]["acer"] <= ranked[1]["acer"]
    assert ranked[0]["acer"] == pytest.approx((ranked[0]["apcer"] + ranked[0]["bpcer"]) / 2)
