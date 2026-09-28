# Campus Video Analysis

A local Flask dashboard for analyzing video streams and reviewing possible
fight activity. The project combines scene-level action classification with
optional person detection and tracking.

> **Status:** Saved-video analysis is the recommended workflow. RTSP monitoring
> is experimental, and detection accuracy on campus footage has not been
> independently validated. Treat alerts as prompts for human review.

## What it does

- Displays analyzed video and synchronized overlays in a browser dashboard.
- Uses an X3D scene classifier to flag **suspected fight activity**.
- Can run YOLO pose detection and ByteTrack tracking to show person boxes.
- Reports stream progress, timestamps, model status, and alert history.

A scene alert describes the video window; it does not identify who is involved
or establish intent. The legacy pose-based LSTM is optional diagnostic code and
is off by default.

## Choose a mode

| Mode | What you get | Recommended for |
|---|---|---|
| `scene_only` | Scene alerts without pose detection or person boxes; uses fewer resources. | Saved-video review |
| `combined` | Scene analysis plus person detection and tracking. | When person boxes are useful |

The default is `combined`. Mode is independent of input type: either mode can be
used with a local video or an RTSP URL. RTSP is experimental and may lose frames
or complete action windows when processing falls behind.

## Requirements

- Windows 64-bit and Python 3.12 are the documented setup.
- The app uses CPU inference; GPU acceleration is not configured or validated.
- A browser with access to the Socket.IO CDN is needed for the dashboard.
- Scene-analysis mode needs the pinned X3D checkpoint. Combined mode also needs
  the bundled YOLO pose model.

The repository currently has no dependency manifest: `requirements.txt` is
absent, and `requirements-runtime.txt` is not present. Install the Python
packages required by the imports in `src/` into your environment before launch.
The app does not download model weights automatically.

## Get the scene model

If `src/models/final_x3d_realtime.pt` is missing, fetch the supported checkpoint
from the repository root with:

```powershell
python tools/fetch_scene_model.py
```

The fetcher verifies the pinned SHA-256 hash before saving the file. The scene
adapter rejects checkpoints that do not match its supported model contract.

## Run saved-video analysis

Open PowerShell at the repository root. Set the mode and launch the dashboard:

```powershell
$env:DETECTION_MODE = 'scene_only'
$env:OFFLINE_REALTIME_PACING = '0'
python src/main.py
```

Then open [http://localhost:5000](http://localhost:5000), add a stream using an
absolute path to a video on the machine running the app, and select **Start**.
Use **Stop** to stop analysis. Start the stream again to reprocess it; changing
the source is not a seek operation.

`OFFLINE_REALTIME_PACING=0` allows local files to process as fast as the system
can manage without a playback-speed cap. The default, `1`, paces decoding toward
the video's source rate when possible. Neither setting guarantees real-time
processing.

To include person boxes, stop the server and restart it in combined mode:

```powershell
$env:DETECTION_MODE = 'combined'
python src/main.py
```

For an RTSP source, launch the app and enter the RTSP URL in the dashboard. Set
`OFFLINE_REALTIME_PACING=1`; that setting applies to local files only. Do not
put camera credentials in tracked configuration or logs.

## Alerts and timestamps

The scene classifier collects about five seconds of source video for each
decision, with a one-second decision stride. By default, two consecutive
positive decisions begin a suspected-fight alert; a valid negative decision
ends it. Inference time and queue delays add to the alert's wall-clock delay,
and brief events may be missed.

| Dashboard state | Meaning |
|---|---|
| Warming up | Collecting enough video for the first complete analysis window. |
| Active | Analysis is running without a current suspected-fight alert. |
| Suspected fight | The scene score met the alert rule; review the video. |
| Unavailable | Model loading or inference failed; check the displayed reason. |
| Observation lost / stale | Fresh observations stopped or reset. The outcome is unknown, not negative. |
| Offline complete | The file ended. This does not establish that a fight ended. |

File timestamps refer to the video timeline. RTSP timestamps are decoder-local
monotonic times, not camera capture timestamps.

## Configuration

Set environment variables before starting the server. Relative paths are
resolved from the repository root unless noted.

| Variable | Default | Description |
|---|---:|---|
| `DETECTION_MODE` | `combined` | `combined` or `scene_only`. |
| `OFFLINE_REALTIME_PACING` | `1` | Set to `0` to remove the local-file playback-speed cap. |
| `SCENE_MODEL_PATH` | `src/models/final_x3d_realtime.pt` | Supported scene checkpoint path. |
| `SCENE_WINDOW_SECONDS` | `5` | Scene history duration in seconds. |
| `SCENE_STRIDE_SECONDS` | `1` | Minimum spacing between decisions in source time. |
| `SCENE_THRESHOLD` | `0.5` | Positive scene-score threshold. |
| `SCENE_CONFIRM_WINDOWS` | `2` | Consecutive positive decisions required to start an alert. |
| `ACTION_THREADS` | `4` | CPU threads used by scene inference. |
| `POSE_MODEL_PATH` | `models/yolov8n-pose_openvino_model` | Pose model path, resolved from `src/`; unused in scene-only mode. |
| `POSE_IMAGE_SIZE` | `320` | Pose model input size. |
| `PERSON_THRESHOLD` | `0.35` | Person detection confidence threshold. |
| `LEGACY_POSE_DIAGNOSTICS` | off | Set to `1` to enable legacy scoring in combined mode. |
| `FIGHT_MODEL_PATH` | `src/models/lstm-violence-detection.onnx` | Legacy LSTM model path. |
| `FIGHT_THRESHOLD` | `0.7` | Legacy diagnostic threshold; not used by scene alerts. |
| `FIGHT_CONTRACT` | unset | Optional legacy model-contract file. |

Thresholds and defaults are implementation settings, not validated operating
points. Do not tune on footage reserved for final evaluation.

## Project layout

```text
src/
  main.py                 Flask server, dashboard routes, stream lifecycle
  worker.py               Video decoding and inference workers
  detector.py             Person/pose detection
  scene_model.py          X3D scene preprocessing and alert windows
  fight_detector_onnx.py  Legacy pose-based ONNX diagnostic
  pipeline_frames.py      Frame geometry and synchronized overlays
  monitor.py              System and pipeline monitoring
  templates/dashboard.html Browser dashboard
  models/                 Scene, pose, and legacy model assets
  training/               Historical model training and conversion code
config/                   Scene model and evaluation contract examples
tests/                    Regression tests
tools/                    Model fetch, evaluation, review, and benchmark tools
```

See [tools/README.md](tools/README.md) for a guide to the utilities. Benchmark
and replay scripts may be resource-intensive and are not needed to launch the
dashboard.

## Model provenance and license

- Project source: [MIT License](LICENSE).
- Scene checkpoint: `visionlab-ai/school-violence-detection-models`, artifact
  `final/final_x3d_realtime.pt`, pinned revision
  `a744b6af7496f0cbfa4f0ba32acd46b65e52d4e1` and SHA-256
  `e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349`.
- The scene model uses 16 sampled frames resized to 224×224, converted from
  BGR to RGB, normalized with mean `0.45` and standard deviation `0.225`.
- Optional Ultralytics pose assets and dependencies have separate license terms;
  the project license does not replace them.

The model predicts scene-level class scores. Five-second causal windows are a
deployment adaptation and do not establish live-stream or campus accuracy.
