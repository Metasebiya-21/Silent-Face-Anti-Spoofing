# Liveness Service

Passive face liveness for a mobile onboarding flow: the customer looks at the phone, nothing else.
Each attempt is a **session**. While the phone records the face, its screen shows a random colour sequence
unique to the session; a worker then checks that the face reflected exactly that sequence (a real face close to
the screen picks up its colour, a photo, replay or pre-recorded video does not). The client only ever learns
`live`, `spoof` or `retry`.

The repo also contains **MiniFASNet**, the single-image anti-spoofing model this project started from
([minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), Apache-2.0),
and the notebooks to retrain it. Fusing its score with the colour challenge is not done yet.

## Run it

```
docker compose up --build             # api on :8000, worker, redis, tusd
```
or locally (Python 3.12, [uv](https://docs.astral.sh/uv/), a Redis and a tusd you run yourself):
```
uv sync --extra api
cp .env.example .env                  # then edit: REDIS_URL, TUSD_URL, TUSD_DATA_DIR, LIVENESS_API_KEY
uv run uvicorn liveness_service.api.http.app:app --port 8000
uv run liveness-worker
uv run pytest                         # sync with --all-extras first for the MiniFASNet and notebook tests
```

tusd must call `POST /hooks/tus` with `-hooks-enabled-events pre-create,post-receive,post-finish,post-terminate`
and share its upload folder with the worker (`TUSD_DATA_DIR`). Keep tusd, Redis and `/hooks/tus` internal: the app
only talks to the API.

## API for the app

| Step | Request | Returns / does |
|---|---|---|
| 1 | `POST /v1/liveness/sessions` (`X-API-Key`, body `{"device_id": ...}`) | `{session_id, token, tus_endpoint, expires_at}` |
| 2 | `GET /v1/liveness/sessions/{id}/events` (`Authorization: Bearer <token>`) | SSE stream, below |
| 3 | start recording on `capture`; show each `colour` on screen as it arrives | |
| 4 | `POST /v1/liveness/sessions/{id}/display-log` | when recording started and when each colour appeared (phone clock) |
| 5 | tus upload to `tus_endpoint`, `Upload-Metadata: session_id ..., token ...` | one upload per session |
| 6 | wait for `done` (or `GET /v1/liveness/sessions/{id}`) | `live` / `spoof` / `retry` |

SSE events, in order: `capture`, `colour` × N (each just before it is due, never earlier), `upload_progress`,
`uploaded`, `analysing`, `done`. Payload models: `src/liveness_service/schemas/outbound/liveness.py`.
Post the display log before the last upload chunk. Expired or finished sessions answer `410 Gone`; a new attempt
needs a new session. `GET /health` for liveness probes; OpenAPI at `/docs`.

## Configuration

Every duration, threshold, palette, TTL and rate limit is a setting in `src/liveness_service/core/settings.py`,
validated at startup. Locally they come from `.env` (see `.env.example`). In production set `VAULT_ADDR`,
`VAULT_TOKEN`, `VAULT_PATH` and `PRODUCTION=true` as environment variables: the rest is read from that HashiCorp
Vault KV v2 secret (keys are the lowercase setting names), and the service refuses to start without
`liveness_api_key` and `session_secret`. Finished sessions are audited (decision, scores, timings, no media) in
SQLite (`AUDIT_DB_PATH`).

**Colour-challenge verifier (learning mode):** the signal processing in
`src/liveness_service/ml/colour_challenge/signals.py` (chromaticity, expected signal, FFT cross-correlation,
per-step match) is left as TODO. `uv run pytest tests/test_colour_signals.py` shows what to implement; until then
the worker answers `retry`. Thresholds are tuned on synthetic data only: recalibrate on real phone captures.

## Layout

```
src/liveness_service/
├── api/http/              FastAPI app and routes (sessions, events, uploads proxy, tus hooks, health)
├── core/                  settings (Vault / .env), security (API key, session tokens), Redis, tusd client
├── services/              sessions, colour sequence, uploads, analysis
├── repositories/          Redis sessions/events/queue, SQLite audit
├── ml/colour_challenge/   face regions (YuNet) + verifier
├── ml/                    MiniFASNet: checker, inference, networks
└── workers/               liveness-worker
src/model_development/     notebooks to train, validate and test MiniFASNet
resources/                 MiniFASNet weights, RetinaFace detector, YuNet face detector
tests/
```

## MiniFASNet

A small anti-spoofing classifier for one face crop: real, or one of two attack types. Two models look at the same
face at different crop scales (2.7× and 4× the face box) and their probabilities are averaged. During training an
auxiliary branch learns the image's Fourier spectrum, which pushes the network towards the texture differences
between real skin and prints or screens.

![architecture](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg)

| Model | FLOPs | Params |
|:---:|:---:|:---:|
| MobileFaceNet | 0.224G | 0.991M |
| MiniFASNetV1 | 0.081G | 0.414M |
| MiniFASNetV2 | 0.081G | 0.435M |

```python
import cv2
from liveness_service.ml.checker import LivenessChecker   # uv sync --extra minifasnet
print(LivenessChecker().check(cv2.imread("images/sample/image_T1.jpg")))  # LivenessResult(...) or None
```
Pixels go in as BGR, 0–255, not normalised. Results depend on the camera and scene; use frames from a real camera
with the whole face visible and turned less than about 30°.

### Retraining (notebooks)

`uv sync --extra train`, then open the notebooks in VS Code or Jupyter from their own folder and run, in order:
1. `src/model_development/training/train.ipynb`: one model for one patch type; checkpoints in `saved_logs/snapshot/`
2. `src/model_development/val/validate.ipynb`: score every checkpoint on the validation split, best by ACER
3. `src/model_development/test/test.ipynb`: final APCER / BPCER / ACER on the test split, optional export to
   `resources/anti_spoof_models/` (every `.pth` there joins the ensemble)

`training/` also holds `config`, `data`, `model` and `trainer`, one function or class per cell, loaded with `%run`.
Data goes in `datasets/rgb_image/{train,val,test}/<patch_info>/{0,1,2}/` (see `datasets/README.md`):
`patch_info` is `<scale>_<height>x<width>`, e.g. `2.7_80x80`, or `org_1_80x60` for the whole image resized.

![patch examples](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)

## Credits

MiniFASNet, its weights and the original training code: [Minivision](https://www.minivision.cn/),
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache-2.0, see `LICENSE`).
Face detectors: [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) (MiniFASNet crops),
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (colour challenge).
