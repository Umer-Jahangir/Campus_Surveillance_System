"""Evaluation helpers; labels describe video time, not every visible person."""

import hashlib
from pathlib import Path


def validate_manifest(clips, root):
    seen_files, seen_sources = {}, {}
    for clip in clips:
        if clip["split"] not in ("tune", "evaluation"):
            raise ValueError("split must be tune or evaluation")
        if not clip.get("source_id"):
            raise ValueError(
                "source_id is required to keep excerpts of one source in one split"
            )
        path = (Path(root) / clip["path"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for key, mapping in ((digest, seen_files), (clip["source_id"], seen_sources)):
            if key in mapping and mapping[key] != clip["split"]:
                raise ValueError(
                    "Tuning/evaluation leakage: same file or source in both splits"
                )
            mapping[key] = clip["split"]
        previous = -1
        for start, end in clip["fight_intervals"]:
            if start < 0 or end <= start or start < previous:
                raise ValueError(
                    "fight_intervals must be ordered, non-overlapping [start,end) seconds"
                )
            previous = end


def metrics(samples, intervals):
    """Frame metrics on processed frames, plus event onsets and delays."""
    tp = fp = fn = tn = false_events = 0
    previous_alarm = False
    delays = [None] * len(intervals)
    for timestamp, predicted in samples:
        positive = any(start <= timestamp < end for start, end in intervals)
        tp += int(positive and predicted)
        fp += int(not positive and predicted)
        fn += int(positive and not predicted)
        tn += int(not positive and not predicted)
        false_events += int(predicted and not positive and not previous_alarm)
        previous_alarm = predicted
        for i, (start, end) in enumerate(intervals):
            if predicted and start <= timestamp < end and delays[i] is None:
                delays[i] = timestamp - start
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_onsets=false_events,
        detection_delay_seconds=delays,
        missed_events=sum(d is None for d in delays),
    )
