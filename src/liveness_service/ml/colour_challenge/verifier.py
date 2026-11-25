"""Colour-challenge verification: did the face reflect exactly this session's colour sequence?

verify_video()   video file -> per-frame colour traces -> verify_traces()
verify_traces()  traces + what was shown -> scores + pass / fail / inconclusive
"""
import cv2
import numpy as np

from liveness_service.exceptions import VideoUnreadableError
from liveness_service.ml.colour_challenge.signals import (
    best_lag_correlation, chromaticity, decide, expected_signal, resample, response_amplitude,
    snr_from_correlation, step_match_rate,
)


def extract_traces(video_path, regions):
    """(t_ms, skin_rgb, background_rgb): one row per frame; NaN rows where no face was found."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoUnreadableError("could not open video")
    times, skin, background = [], [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))  # container timestamp: correct for variable frame rate
            s, b = regions.measure(frame)
            skin.append(s if s is not None else (np.nan,) * 3)
            background.append(b if b is not None else (np.nan,) * 3)
    finally:
        cap.release()
    if not times:
        raise VideoUnreadableError("video has no frames")
    return np.asarray(times, float), np.asarray(skin, float), np.asarray(background, float)


def verify_traces(t_ms, skin_rgb, background_rgb, shown, config):
    """shown: list of ShownStep (what was on screen, when). Returns (outcome, scores)."""
    face = chromaticity(skin_rgb)
    background = chromaticity(background_rgb)
    has_face = ~np.isnan(face).any(axis=1)
    # only the part of the recording where colours were on screen counts
    start = shown[0].shown_at_ms
    window = (t_ms >= start) & (t_ms <= shown[-1].shown_at_ms + 1000)
    scores = {"frames": int(window.sum()), "face_frame_ratio": float(has_face[window].mean()) if window.any() else 0.0}
    if scores["face_frame_ratio"] == 0:
        return decide({**scores, "face_amplitude": 0.0}, config), scores

    grid = np.arange(start, t_ms[window & has_face].max(), 1000 / config.resample_hz)
    face_u = resample(t_ms[window], face[window], grid)
    expected_u = expected_signal(grid, shown)
    scores["face_amplitude"] = response_amplitude(face_u)
    background_amplitude = response_amplitude(background[window])
    scores["face_bg_ratio"] = (scores["face_amplitude"] / background_amplitude if background_amplitude > 0
                               else float("inf"))
    lag_ms, correlation = best_lag_correlation(face_u, expected_u, config.resample_hz, config.max_lag_ms)
    scores.update(lag_ms=float(lag_ms), correlation=float(correlation), snr=snr_from_correlation(correlation),
                  step_match_rate=float(step_match_rate(t_ms, face, shown, lag_ms, config.step_window_ms)))
    return decide(scores, config), scores


def verify_video(video_path, shown, config, regions):
    return verify_traces(*extract_traces(video_path, regions), shown, config)
