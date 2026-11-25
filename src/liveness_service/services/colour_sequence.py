"""Random colour sequence for one session's colour challenge. Unpredictable: every choice comes from `secrets`."""
import secrets
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ColourStep:
    index: int
    colour: str               # palette name
    rgb: tuple[int, int, int]  # what the phone puts on screen
    duration_ms: int
    offset_ms: int            # planned start, relative to the first step

    def to_dict(self):
        return asdict(self)


def generate(palette, length, min_ms, max_ms, max_changes_per_second):
    """palette: {name: (r, g, b)}. Rules: never the same colour twice in a row; each step lasts min_ms..max_ms;
    at most max_changes_per_second colour changes in any 1 s window."""
    if len(palette) < 2 or min_ms > max_ms or (max_changes_per_second - 1) * min_ms + max_ms <= 1000:
        raise ValueError("colour sequence rules can't satisfy each other")  # Settings validates the same
    names = sorted(palette)
    steps, offset = [], 0
    for index in range(length):
        colour = secrets.choice([n for n in names if not steps or n != steps[-1].colour])
        # Colour changes happen at step starts. "At most k per second" means any k+1 consecutive starts span
        # more than 1000 ms: this step plus the k-1 before it must last > 1000 ms in total.
        k = max_changes_per_second
        previous = sum(s.duration_ms for s in steps[len(steps) - (k - 1):]) if k > 1 else 0
        low = max(min_ms, 1001 - previous) if index >= k - 1 else min_ms
        duration = low + secrets.randbelow(max_ms - low + 1)
        steps.append(ColourStep(index, colour, tuple(palette[colour]), duration, offset))
        offset += duration
    return steps


def from_dicts(items):
    return [ColourStep(**{**item, "rgb": tuple(item["rgb"])}) for item in items]
