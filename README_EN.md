[中文版](README.md)|**English Version**  
![Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  
# Silent-Face-Anti-Spoofing 

This project is Silent-Face-Anti-Spoofing belongs to [minivision technology](https://www.minivision.cn/). You can scan the QR code below to get APK and install it on Android side to experience the effect of real time living detection(silent face anti-spoofing detection).   
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/静默活体APK.jpeg" width="200" height="200" align=center />  

## Introduction

In this project, we open source the silent face anti-spoofing model with training architecture, data preprocessing method, model training & test script and open source APK for real time testing.  

The main purpose of silent face anti-spoofing detection technology is to judge whether the face in front of the machine is real or fake. The face presented by other media can be defined as false face, including printed paper photos, display screen of electronic products, silicone mask, 3D human image, etc. At present, the mainstream solutions includes cooperative living detection and non cooperative living detection (silent living detection). Cooperative living detection requires the user to complete the specified action according to the prompt, and then carry out the live verification, while the silent live detection directly performs the live verification.  

Since the Fourier spectrum can reflect the difference of true and false faces in frequency domain to a certain extent, we adopt a silent living detection method based on the auxiliary supervision of Fourier spectrum. The model architecture consists of the main classification branch and the auxiliary supervision branch of Fourier spectrum. The overall architecture is shown in the following figure:

![overall architecture](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg)  

By using our self-developed model pruning method, the FLOPs of MobileFaceNet is reduced from 0.224G to 0.081G, and the performance of the model is significantly improved (the amount of calculation and parameters is reduced) with little loss of precision.


|Model|FLOPs|Params|
| :------:|:-----:|:-----:| 
|MobileFaceNet|0.224G|0.991M|
|MiniFASNetV1|0.081G|0.414M|
|MiniFASNetV2|0.081G|0.435M|

## APK
### APK source code  
Open source for Android platform deployment code: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK  

### Demo
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif" width="300" height="400"/>  
 
### Performance  
| Model|FLOPs|Speed| FPR | TPR |comments |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK |84M| 20ms | 1e-5|97.8%| Open Source|
| High precision model |162M| 40ms| 1e-5 |99.7%| Private |

### Test Method 

- Display information: speed(ms), confidence(0 ~ 1) and in living detection test results (true face or false face).
- Click the icon in the upper right corner to set the threshold value. If the confidence level is greater than the threshold value, it is a true face, otherwise it is a fake face.

### Before test you must know

- All the test images must be collected by camera, otherwise it does not conform to the normal scene usage specification, and the algorithm effect cannot be guaranteed.
- Because the robustness of RGB silent living detection depending on camera model and scene, the actual use experience could be different.
- During the test, it should be ensured that a complete face appears in the view, and the rotation angle and vertical direction of the face are less than 30 degrees (in line with the normal face recognition scene), otherwise, the experience will be affected.　

**Tested mobile phone processor**

|type|Kirin990 5G|Kirin990 |Qualcomm845 |Kirin810 |RK3288 |
| :------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Speed/ms|19|23|24|25|90|

## Repo
### Install dependency Library  
```
uv sync                  # inference only; add `--extra train` for training
```
### Clone
```
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing  
cd Silent-Face-Anti-Spoofing
```  
### Data Preprocessing
1.The training set is divided into three categories, and the pictures of the same category are put into a folder;  
2.Due to the multi-scale model fusion method, the original image and different patch are used to train the model, so the data is divided into the original map and the patch based on the Original picture;  
- Original picture(org_1_height**x**width),resize the original image to a fixed size (width, height), as shown in Figure 1;  
- Patch based on original(scale_height**x**width),The face detector is used to obtain the face frame, and the edge of the face frame is expanded according to a certain scale. In order to ensure the consistency of the input size of the model, the area of the face frame is resized to a fixed size (width, height). Fig. 2-4 shows the patch examples with scales of 1, 2.7 and 4;  
![patch demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)  

3.The Fourier spectrum is used as the auxiliary supervision, and the corresponding Fourier spectrum is generated online from the training set images.  
**The directory structure of the dataset is shown below**
```
├── datasets
    └── RGB_Images
        ├── org_1_80x60
            ├── 0
		├── aaa.png
		├── bbb.png
		└── ...
            ├── 1
		├── ddd.png
		├── eee.png
		└── ...
            └── 2
		├── ggg.png
		├── hhh.png
		└── ...
        ├── 1_80x80
        └── ...
```  
### Train, validate, test (notebooks)
Model development is notebooks only, in `src/model_development/` (open them in VS Code or Jupyter after
`uv sync --extra train`, from their own folder). Each function or class has its own cell; notebooks load each other
with `%run`. Run these three, in order:
1. `training/train.ipynb`: train one model for one patch type; checkpoints go to `saved_logs/snapshot/`
2. `val/validate.ipynb`: score every checkpoint on the validation split, pick the best by ACER
3. `test/test.ipynb`: final APCER / BPCER / ACER on the test split, optional export to `resources/anti_spoof_models/`

The other notebooks hold the code they run: `training/` has `config`, `data` (dataset, Fourier target, torchvision augmentation), `model` (MultiFTNet)
and `trainer`; `val/evaluation.ipynb` has the metrics (also used by `test.ipynb`).

Data is split into `datasets/rgb_image/{train,val,test}/<patch_info>/{0,1,2}/` (see `datasets/README.md`).
### Test
 ./resources/anti_spoof_models Fusion model of in living detection  
 ./resources/detection_model Detector  
 ./images/sample Test Images  
 ```python
 import cv2
 from liveness_service.ml.checker import LivenessChecker   # uv sync --extra minifasnet
 print(LivenessChecker().check(cv2.imread("images/sample/image_T1.jpg")))  # LivenessResult(is_real=..., score=...) or None
 ```    
### Liveness service (sessions + colour challenge)
Each capture attempt is a **session**. While the phone records the customer's face, its screen shows a random
colour sequence unique to the session; afterwards a worker checks that the face reflected exactly that sequence.

```
docker compose up --build            # api :8000, worker, redis, tusd
uv run uvicorn liveness_service.api.http.app:app --port 8000   # or locally: cp .env.example .env first
uv run liveness-worker
```

The app only talks to `/v1/liveness`:
| Step | Request | |
|---|---|---|
| 1 | `POST /v1/liveness/sessions` (`X-API-Key`, body `{"device_id"}`) | `{session_id, token, tus_endpoint, expires_at}` |
| 2 | `GET /v1/liveness/sessions/{id}/events` (`Authorization: Bearer <token>`) | SSE, see below |
| 3 | show each `colour` event on screen as it arrives; start recording on `capture` | |
| 4 | `POST /v1/liveness/sessions/{id}/display-log` | phone-clock times each colour appeared, and when recording started |
| 5 | tus upload to `tus_endpoint`, `Upload-Metadata: session_id ..., token ...` | single use: one upload per session |
| 6 | wait for `done` on the stream (or `GET /v1/liveness/sessions/{id}`) | `live` / `spoof` / `retry` |

SSE events: `capture` (start recording), `colour` x N (just in time, never earlier), `upload_progress`, `uploaded`,
`analysing`, `done` (decision only). Payload models: `src/liveness_service/schemas/outbound/liveness.py`.
Expired or finished sessions answer `410 Gone`; a new attempt needs a new session.

tusd must call `POST /hooks/tus` with events `pre-create,post-receive,post-finish,post-terminate`, and share its
upload folder with the worker (`TUSD_DATA_DIR`). Keep tusd and `/hooks/tus` internal.

Every duration, threshold, palette and TTL is a setting (`src/liveness_service/core/settings.py`), from Vault in
production (`VAULT_ADDR`/`VAULT_TOKEN`/`VAULT_PATH`, `PRODUCTION=true`) or `.env` locally. Decisions are audited
(no media) in SQLite (`AUDIT_DB_PATH`).

**Learning mode:** the colour-challenge signal processing (`ml/colour_challenge/signals.py`: chromaticity,
expected signal, FFT cross-correlation, per-step match) is left as TODO; `uv run pytest tests/test_colour_signals.py`
shows what to implement. Until then the worker answers `retry`.

## Reference 
- Detector [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)  

For this project, in order to facilitate the technical exchange of developers, we created QQ group: 1121178835, welcome to join.  

In addition to the open-source silent living detection algorithm, Minivision technology also has a number of self-developed algorithms and SDK related to face recognition and human body recognition. Interested individual developers or enterprise developers can visit our website: [Mini-AI Open Platform](https://ai.minivision.cn/)
Welcome to contact us.
