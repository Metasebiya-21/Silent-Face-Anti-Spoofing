"""Colour-challenge signal processing.

LEARNING MODE: the tests marked TODO fail until you implement the matching function in
src/liveness_service/ml/colour_challenge/signals.py. Run just these with:
    uv run pytest tests/test_colour_signals.py -q
Work top to bottom: each function builds on the ones above it.
"""
import numpy as np
import pytest

from liveness_service.enums import CheckOutcome
from liveness_service.ml.colour_challenge.signals import (
    ShownStep, VerifierConfig, best_lag_correlation, chromaticity, expected_signal, step_match_rate,
)
from liveness_service.ml.colour_challenge.verifier import verify_traces
from tests.synthetic import BACKGROUND, SEQUENCE_A, SEQUENCE_B, SKIN, shown, simulate

CONFIG = VerifierConfig(resample_hz=30, max_lag_ms=300, step_window_ms=150, min_face_frame_ratio=0.8,
                        min_response_amplitude=0.002, pass_min_correlation=0.5, pass_min_step_match=0.6,
                        pass_min_face_bg_ratio=1.5)
THIRD = 1 / 3


# --- TODO 1/4: chromaticity --------------------------------------------------------------------------------
# Hint: divide each row by its own sum. keepdims=True on the sum keeps the shapes broadcastable.

def test_chromaticity_removes_brightness():
    out = chromaticity(np.array([[100.0, 100.0, 100.0], [2.0, 1.0, 1.0], [200.0, 100.0, 100.0]]))
    np.testing.assert_allclose(out, [[THIRD, THIRD, THIRD], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])


def test_chromaticity_black_and_missing_rows_are_nan():
    # Hint: a 0 sum would divide by zero; NaN in means NaN out. np.errstate silences the warnings.
    out = chromaticity(np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [1.0, 1.0, 2.0]]))
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2], [0.25, 0.25, 0.5])


# --- TODO 2/4: expected_signal ------------------------------------------------------------------------------
# Hint: np.searchsorted(starts, t, side="right") - 1 gives, for every t, the index of the step on screen.

def test_expected_signal_holds_each_colour_until_the_next():
    steps = [ShownStep(0, (255, 255, 255)), ShownStep(500, (0, 0, 255))]
    out = expected_signal(np.array([-10.0, 0.0, 499.0, 500.0, 900.0]), steps)
    assert np.isnan(out[0]).all()                       # before the first colour
    np.testing.assert_allclose(out[1:3], [[THIRD] * 3] * 2)  # white
    np.testing.assert_allclose(out[3:], [[0, 0, 1]] * 2)     # blue, held


# --- TODO 3/4: best_lag_correlation -------------------------------------------------------------------------
# Hint: scipy.signal.correlation_lags(n, n) tells you which lag each entry of correlate(..., "full") is.
# Divide by the number of overlapping samples (n - lag) so long and short lags are comparable.

def _steps_signal(n=150, seed=1):
    rng = np.random.default_rng(seed)
    return np.repeat(rng.random((n // 10, 3)), 10, axis=0)  # piecewise-constant, like a colour sequence


def test_best_lag_finds_the_delay():
    expected = _steps_signal()
    observed = np.roll(expected, 3, axis=0)  # 3 samples late at 30 Hz = 100 ms
    observed += np.random.default_rng(2).normal(0, 0.01, observed.shape)
    lag_ms, corr = best_lag_correlation(observed, expected, rate_hz=30, max_lag_ms=300)
    assert lag_ms == pytest.approx(100, abs=1)
    assert corr > 0.9


def test_best_lag_correlation_is_scale_free_and_signed():
    expected = _steps_signal()
    _, corr = best_lag_correlation(expected * 0.001 + 5, expected, rate_hz=30, max_lag_ms=300)
    assert corr == pytest.approx(1, abs=0.02)   # tiny but perfectly following: still ~1
    _, corr = best_lag_correlation(-expected, expected, rate_hz=30, max_lag_ms=0)
    assert corr == pytest.approx(-1, abs=0.02)  # moving the opposite way


def test_best_lag_skips_flat_channels():
    expected = _steps_signal()
    expected[:, 0] = 0.3  # a channel that never changes has std 0
    lag_ms, corr = best_lag_correlation(expected.copy(), expected, rate_hz=30, max_lag_ms=300)
    assert lag_ms == 0 and corr == pytest.approx(1, abs=0.01)


# --- TODO 4/4: step_match_rate ------------------------------------------------------------------------------
# Hint: boolean masks on t_ms pick the frames in each window; np.dot(delta_observed, delta_expected) > 0.

def test_step_match_rate_counts_followed_changes():
    steps = shown(SEQUENCE_A)
    t, skin, _ = simulate(steps, lag_ms=80, noise=0.2)
    face = chromaticity(skin)
    assert step_match_rate(t, face, steps, lag_ms=80, window_ms=150) == pytest.approx(1.0)
    # a face whose colour moves the opposite way follows none of the changes
    inverted = 2 * np.nanmean(face, axis=0) - face
    assert step_match_rate(t, inverted, steps, lag_ms=80, window_ms=150) == pytest.approx(0.0)


# --- the whole verifier, on synthetic captures (pass once the four TODOs are done) ------------------------

def test_real_face_following_the_sequence_passes():
    steps = shown(SEQUENCE_A)
    outcome, scores = verify_traces(*simulate(steps, lag_ms=80), steps, CONFIG)
    assert outcome == CheckOutcome.PASS, scores
    assert scores["lag_ms"] == pytest.approx(80, abs=34)  # within one sample at 30 Hz
    assert scores["face_bg_ratio"] > 1.5


def test_random_colour_fails():
    steps = shown(SEQUENCE_A)
    t, _, background = simulate(steps)
    random_skin = SKIN + np.random.default_rng(3).normal(0, 15, (len(t), 3))
    outcome, scores = verify_traces(t, random_skin, background, steps, CONFIG)
    assert outcome == CheckOutcome.FAIL, scores


def test_tiny_response_is_inconclusive_not_a_spoof():
    steps = shown(SEQUENCE_A)
    outcome, scores = verify_traces(*simulate(steps, face_gain=0.0005, background_gain=0.0001, noise=0.05),
                                    steps, CONFIG)
    assert outcome == CheckOutcome.INCONCLUSIVE, scores


def test_another_sessions_sequence_fails():
    recorded = simulate(shown(SEQUENCE_A))           # the video reflects session A's colours...
    outcome, scores = verify_traces(*recorded, shown(SEQUENCE_B), CONFIG)  # ...but is checked against B's
    assert outcome == CheckOutcome.FAIL, scores


def test_flat_photo_reflecting_like_its_background_fails():
    steps = shown(SEQUENCE_A)
    t, skin, _ = simulate(steps, face_gain=0.15)
    _, background, _ = simulate(steps, face_gain=0.15, seed=4)  # the background reacts as much as the "face"
    outcome, scores = verify_traces(t, skin, background - SKIN + BACKGROUND, steps, CONFIG)
    assert outcome == CheckOutcome.FAIL and scores["face_bg_ratio"] < 1.5, scores


def test_no_face_in_most_frames_is_inconclusive():
    steps = shown(SEQUENCE_A)
    t, skin, background = simulate(steps)
    skin[::2] = np.nan  # face found in only half the frames
    outcome, _ = verify_traces(t, skin, background, steps, CONFIG)
    assert outcome == CheckOutcome.INCONCLUSIVE
