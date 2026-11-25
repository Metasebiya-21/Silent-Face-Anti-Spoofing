"""Synthetic captures for the colour-challenge tests: a face (and background) lit partly by the phone screen."""
import numpy as np

from liveness_service.ml.colour_challenge.signals import ShownStep

SKIN = np.array([180.0, 130.0, 110.0])        # ambient-lit skin, mean RGB
BACKGROUND = np.array([110.0, 115.0, 120.0])  # wall behind the person

# two sessions' worth of (colour, duration_ms); B plays the role of "another session's sequence"
PALETTE = {"white": (255, 255, 255), "blue": (0, 0, 255), "green": (0, 255, 0),
           "yellow": (255, 255, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255)}
SEQUENCE_A = [("white", 400), ("blue", 350), ("green", 500), ("yellow", 450), ("magenta", 300),
              ("cyan", 550), ("blue", 400), ("white", 350), ("green", 500), ("magenta", 450)]
SEQUENCE_B = [("cyan", 450), ("yellow", 400), ("white", 500), ("magenta", 350), ("green", 550),
              ("blue", 300), ("yellow", 450), ("cyan", 400), ("white", 500), ("blue", 350)]


def shown(sequence, start_ms=500):
    """ShownStep list for a (colour, duration) sequence starting at start_ms of the recording."""
    steps, t = [], start_ms
    for colour, duration in sequence:
        steps.append(ShownStep(shown_at_ms=t, rgb=PALETTE[colour]))
        t += duration
    return steps


def screen_rgb(t_ms, steps):
    """RGB on screen at each time (black before the first step and after the last one ends)."""
    out = np.zeros((len(t_ms), 3))
    for i, step in enumerate(steps):
        end = steps[i + 1].shown_at_ms if i + 1 < len(steps) else step.shown_at_ms + 400
        out[(t_ms >= step.shown_at_ms) & (t_ms < end)] = step.rgb
    return out


def simulate(steps, lag_ms=80, face_gain=0.15, background_gain=0.03, noise=1.0, fps=30, seed=0):
    """(t_ms, skin_rgb, background_rgb) of a face reacting to `steps` lag_ms late, with camera noise.
    gain = share of the screen's light the region picks up: a real face near the screen picks up much more
    than the wall behind it."""
    rng = np.random.default_rng(seed)
    end = steps[-1].shown_at_ms + 900
    t = np.arange(0, end, 1000 / fps)
    light = screen_rgb(t - lag_ms, steps)
    skin = SKIN + face_gain * light + rng.normal(0, noise, (len(t), 3))
    background = BACKGROUND + background_gain * light + rng.normal(0, noise, (len(t), 3))
    return t, skin, background
