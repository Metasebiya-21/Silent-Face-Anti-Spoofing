"""Signal processing for the colour challenge.

The idea: the phone screen lights the face. When the screen turns blue, a real face close to it gets a little
bluer, a few tens of ms later (camera + display latency). We compare HOW the face colour moves over time
(observed) with how the screen colour moved (expected).

Conventions used by every function here:
- times are milliseconds relative to the start of the recording (video t=0), as float arrays;
- colour traces are (N, 3) float arrays, columns R, G, B;
- "chromaticity" means colour with brightness removed: each channel divided by R+G+B, so rows sum to 1.

LEARNING MODE: the four functions marked TODO are yours. tests/test_colour_signals.py has a failing test for
each, with hints. Everything else here is implemented and uses them.
"""
from dataclasses import dataclass

import numpy as np

from liveness_service.enums import CheckOutcome


@dataclass(frozen=True)
class ShownStep:
    """One colour as it was actually on screen."""
    shown_at_ms: float       # relative to recording start (from the display log)
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class VerifierConfig:
    resample_hz: int
    max_lag_ms: int
    step_window_ms: int
    min_face_frame_ratio: float
    min_response_amplitude: float
    pass_min_correlation: float
    pass_min_step_match: float
    pass_min_face_bg_ratio: float


# ---------------------------------------------------------------------------------------------------------
# TODO 1/4
def chromaticity(rgb):
    """Convert mean RGB per frame to chromaticity (brightness removed).

    rgb: (N, 3) array of mean R, G, B per frame (any scale, e.g. 0..255). May contain NaN rows (no face).
    Returns (N, 3): each row divided by its own sum, so every valid row sums to 1.
    Rows that are NaN, or whose sum is 0 (pure black, no light), must come out as NaN rows.

    Why: a face getting brighter overall (someone turned on a lamp) must not look like a colour change.
    """
    raise NotImplementedError("TODO 1/4: chromaticity")


# TODO 2/4
def expected_signal(t_ms, shown):
    """What the face 'should' look like: the screen's chromaticity at each time in t_ms.

    t_ms: (M,) sample times. shown: list of ShownStep sorted by shown_at_ms.
    Returns (M, 3): for each t, the chromaticity of the last step with shown_at_ms <= t
    (a step function that holds each colour until the next one). Before the first step: NaN.

    Reuse chromaticity() for the step colours.
    """
    raise NotImplementedError("TODO 2/4: expected_signal")


# TODO 3/4
def best_lag_correlation(observed, expected, rate_hz, max_lag_ms):
    """Find how late the face follows the screen, and how well.

    observed, expected: (N, 3) chromaticity on the same uniform time grid (rate_hz samples per second, no NaN).
    Search lags from 0 to max_lag_ms, where lag L means observed[n + L] lines up with expected[n]
    (the face reacts AFTER the screen changes).
    Returns (lag_ms, correlation): the lag with the highest correlation, and that correlation, in [-1, 1].

    Use scipy.signal.correlate(..., method="fft") per channel. Standardise each channel first (subtract the
    mean, divide by the std) so the result is a correlation coefficient, not a raw dot product; skip channels
    whose std is 0; average the per-channel correlations at each lag.
    """
    raise NotImplementedError("TODO 3/4: best_lag_correlation")


# TODO 4/4
def step_match_rate(t_ms, observed, shown, lag_ms, window_ms):
    """Share of colour changes the face actually followed.

    t_ms: (N,) frame times; observed: (N, 3) face chromaticity (may contain NaN rows); shown: list of ShownStep;
    lag_ms: the lag found by best_lag_correlation; window_ms: how much time to average on each side.
    For every change (step i >= 1, at shown[i].shown_at_ms + lag_ms):
      - delta_observed = mean(observed in the window after the change) - mean(observed in the window before)
      - delta_expected = chromaticity(shown[i].rgb) - chromaticity(shown[i-1].rgb)
      - it matches if the two deltas point the same way: their dot product is > 0.
    Skip changes where either window has no valid frames. Returns matches / changes checked (0.0 if none).
    """
    raise NotImplementedError("TODO 4/4: step_match_rate")


# ---------------------------------------------------------------------------------------------------------
# implemented

def resample(t_ms, values, grid_ms):
    """Linear interpolation of (N, 3) values (NaN rows dropped) onto the uniform grid_ms."""
    valid = ~np.isnan(values).any(axis=1)
    return np.column_stack([np.interp(grid_ms, t_ms[valid], values[valid, c]) for c in range(values.shape[1])])


def response_amplitude(chroma):
    """How much the colour moves at all: mean over channels of the std over time (NaN rows ignored)."""
    return float(np.nanmean(np.nanstd(chroma, axis=0)))


def snr_from_correlation(correlation):
    """Share of variance explained by the screen vs the rest: r² / (1 - r²)."""
    r2 = min(max(correlation, 0.0) ** 2, 0.999999)
    return r2 / (1 - r2)


def decide(scores, config):
    """pass / fail / inconclusive from the scores (dict built by verifier.verify_traces)."""
    if scores["face_frame_ratio"] < config.min_face_frame_ratio:
        return CheckOutcome.INCONCLUSIVE   # face mostly not visible: ask again rather than call it a spoof
    if scores["face_amplitude"] < config.min_response_amplitude:
        return CheckOutcome.INCONCLUSIVE   # too little colour movement to judge (e.g. bright ambient light)
    if (scores["correlation"] >= config.pass_min_correlation
            and scores["step_match_rate"] >= config.pass_min_step_match
            and scores["face_bg_ratio"] >= config.pass_min_face_bg_ratio):
        return CheckOutcome.PASS
    return CheckOutcome.FAIL
