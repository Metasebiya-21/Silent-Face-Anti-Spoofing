import random
import secrets

import pytest
from pydantic import ValidationError

from liveness_service.core.settings import Settings
from liveness_service.services import colour_sequence

PALETTE = Settings().colour_palette


def generate(**overrides):
    args = dict(palette=PALETTE, length=10, min_ms=300, max_ms=600, max_changes_per_second=3) | overrides
    return colour_sequence.generate(**args)


@pytest.mark.parametrize("run", range(200))
def test_sequence_rules(run):
    steps = generate()
    assert len(steps) == 10 and [s.index for s in steps] == list(range(10))
    assert all(a.colour != b.colour for a, b in zip(steps, steps[1:]))            # no colour twice in a row
    assert all(300 <= s.duration_ms <= 600 for s in steps)                       # duration in range
    assert all(s.colour in PALETTE and s.rgb == PALETTE[s.colour] for s in steps)
    assert all(b.offset_ms == a.offset_ms + a.duration_ms for a, b in zip(steps, steps[1:]))
    # at most 3 changes in any 1 s window: any 4 consecutive change instants span more than 1000 ms
    starts = [s.offset_ms for s in steps]
    assert all(starts[i + 3] - starts[i] > 1000 for i in range(len(starts) - 3))


def test_rate_rule_holds_when_it_forces_longer_steps():
    # 2 changes per second with 450..700 ms steps: after a short step, the next must be lengthened
    for _ in range(200):
        starts = [s.offset_ms for s in generate(min_ms=450, max_ms=700, max_changes_per_second=2, length=20)]
        assert all(starts[i + 2] - starts[i] > 1000 for i in range(len(starts) - 2))


def test_unsatisfiable_rules_are_refused():
    with pytest.raises(ValueError, match="can't satisfy"):
        generate(min_ms=200, max_ms=700, max_changes_per_second=2)  # 200 + 700 <= 1000


def test_uses_secrets_never_random(monkeypatch):
    calls = []
    monkeypatch.setattr(secrets, "choice", lambda seq: calls.append("choice") or seq[0])
    monkeypatch.setattr(secrets, "randbelow", lambda n: calls.append("randbelow") or 0)
    monkeypatch.setattr(random, "choice", lambda *a: pytest.fail("random.choice used"))
    monkeypatch.setattr(random, "randint", lambda *a: pytest.fail("random.randint used"))
    generate(length=5)
    assert calls.count("choice") == 5 and calls.count("randbelow") == 5
    assert "random" not in vars(colour_sequence)


def test_sequences_differ_between_sessions():
    assert len({tuple((s.colour, s.duration_ms) for s in generate()) for _ in range(20)}) == 20


@pytest.mark.parametrize("env", [
    {"COLOUR_PALETTE": '{"white": [255, 255, 255]}'},                               # one colour can't alternate
    {"COLOUR_STEP_MIN_MS": "700", "COLOUR_STEP_MAX_MS": "600"},
    {"COLOUR_STEP_MIN_MS": "100", "COLOUR_STEP_MAX_MS": "300", "COLOUR_MAX_CHANGES_PER_SECOND": "3"},  # 100+100+300
    {"COLOUR_PALETTE": '{"white": [255, 255, 256], "blue": [0, 0, 255]}'},          # not a colour
])
def test_unsatisfiable_colour_settings_are_rejected(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_production_needs_a_real_session_secret(monkeypatch):
    monkeypatch.setenv("PRODUCTION", "true")
    monkeypatch.setenv("LIVENESS_API_KEY", "k")
    with pytest.raises(ValidationError, match="session_secret"):
        Settings()
    monkeypatch.setenv("SESSION_SECRET", "a-real-secret")
    assert Settings().production
