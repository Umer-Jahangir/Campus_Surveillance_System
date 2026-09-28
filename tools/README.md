# Utility index

Run utilities from the repository root. Current setup and usage live in the
[main README](../README.md).

| Group | Files | Purpose |
|---|---|---|
| Setup | `fetch_scene_model.py` | Download and hash-check the pinned X3D checkpoint. |
| Review | `review_clips.py`, `prepare_validation_review.py`, `demo_scene.py` | Historical excerpts and scene demonstrations; inspect local source paths first. |
| Legacy evaluation | `evaluate.py` | Legacy pose-contract evaluator, **not** an X3D accuracy command. |
| Operational measurements | `benchmark_scene.py`, `benchmark_rtsp.py`, `focused_validate.py` | Resource-intensive load/replay tools, not routine smoke checks. Some require local FFmpeg/MediaMTX paths. |
| Result summaries | `summarize_load.py`, `summarize_rtsp.py`, `summarize_focused.py` | Recalculate summaries from saved measurements. |
| Walkthrough production | `build_*.ps1`, `rebuild_*.ps1`, `update_*.ps1` | Existing presentation/video tooling with local dependencies; preserved as unrelated work. |

The short integration helper remains at
[`audit/scene_api_smoke.py`](../audit/scene_api_smoke.py) to preserve historical
references. The root README gives explicit source/output arguments that avoid
overwriting evidence. Do not run benchmarks to verify documentation or formatting.
