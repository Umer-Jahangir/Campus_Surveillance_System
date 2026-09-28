"""Event metrics for completely human-labelled source-time intervals.

An alert matches at most one fight and each fight at most one alert. Overlap is
required (no post-event grace). Duplicate alerts are false alerts. Unknown EOF
outcomes truncate observation, not the physical ground-truth event. All footage,
including warmup, belongs in the denominator. These choices must be frozen
before independent evaluation.
"""

import math


def event_metrics(duration, fights, alerts, *, confirmed=False):
    if not confirmed:
        raise ValueError(
            "Confirmed complete human annotations required; provisional/unknown is not negative"
        )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    for intervals in (fights, alerts):
        previous = 0
        for start, end in intervals:
            if (
                not all(map(math.isfinite, (start, end)))
                or not 0 <= start < end <= duration
                or start < previous
            ):
                raise ValueError(
                    "Intervals must be ordered, disjoint and within duration"
                )
            previous = end
    # Ordered disjoint intervals permit earliest-overlap matching. One long
    # alert cannot count as detection of arbitrarily many distinct fights.
    matches = {}
    for ai, (a, b) in enumerate(alerts):
        for fi, (s, e) in enumerate(fights):
            if fi not in matches and min(b, e) > max(a, s):
                matches[fi] = ai
                break
    tp = len(matches)
    fp = len(alerts) - tp
    fn = len(fights) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "duration_seconds": duration,
        "camera_hours": duration / 3600,
        "fight_events": len(fights),
        "alert_events": len(alerts),
        "matched_events": tp,
        "false_alerts": fp,
        "missed_fights": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alerts_per_camera_hour": fp / (duration / 3600),
        "time_to_first_alert_seconds": [
            max(0, alerts[matches[i]][0] - s) if i in matches else None
            for i, (s, e) in enumerate(fights)
        ],
        "matching": "one-to-one temporal overlap, no grace; source-time delay; pre-existing alert overlap has zero delay",
    }
