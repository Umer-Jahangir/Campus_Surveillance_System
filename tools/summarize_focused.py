"""Summarize saved focused runs without loading inference dependencies."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit/validation-20260918/focused'


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * p
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (position-lo)


def summarize(data):
    samples = data['samples']
    frames = [f for f in data['frames'] if f['elapsed'] >= 15]
    decisions = {}
    for frame in frames:
        action = frame['action']
        if action.get('window_end') is not None:
            decisions.setdefault(action['window_end'], frame)
    windows = list(decisions.values())
    result = {k: data[k] for k in ['mode','requested_seconds','observed_seconds','failure']}
    if samples:
        result.update(peak_rss_mib=max(s['rss_mib'] for s in samples),
                      min_available_mib=min(s['available_mib'] for s in samples),
                      first_10_rss_mib=mean(s['rss_mib'] for s in samples[:10]),
                      last_10_rss_mib=mean(s['rss_mib'] for s in samples[-10:]),
                      mean_host_cpu_percent=mean(s['host_cpu_percent'] for s in samples),
                      mean_app_cpu_one_core_percent=mean(s['app_cpu_one_core_percent'] for s in samples[1:] or samples),
                      max_frame_queue=max(max(s['frame_queues'], default=0) for s in samples),
                      max_scene_queue=max(max(s['scene_queues'], default=0) for s in samples))
    if len(frames) >= 2:
        span = frames[-1]['elapsed'] - frames[0]['elapsed']
        processed = frames[-1]['stats']['frames'] - frames[0]['stats']['frames']
        source_frames = frames[-1]['frame_id'] - frames[0]['frame_id']
        result.update(processed_fps=processed/span, decoded_frame_id_fps=source_frames/span,
                      skipped_frames_in_interval=source_frames-processed,
                      unique_windows=len(windows), decisions_per_minute=len(windows)/span*60,
                      max_source_decision_gap_s=max((f['action'].get('decision_gap_seconds') or 0 for f in windows), default=0),
                      max_decision_receipt_gap_s=max((b['elapsed']-a['elapsed'] for a,b in zip(windows,windows[1:])), default=0),
                      model_instances=frames[-1]['profile'],
                      events_by_phase={phase:sum(e['phase']==phase for e in data['events'])
                                       for phase in {e['phase'] for e in data['events']}})
        for key in ['receipt_ms','queue_ms','delivery_ms','pipeline_ms']:
            result[key] = {q: percentile([f.get(key) for f in frames],p) for q,p in [('p50',.5),('p95',.95)]}
        for key in ['decode_ms','prepare_ms','pose_ms','jpeg_ms']:
            result[key+'_mean'] = mean(f['profile'][key] for f in frames)
        for key in ['preprocess_ms','action_ms']:
            result[key+'_mean'] = mean(f['action'][key] for f in windows) if windows else None
    return result


if __name__ == '__main__':
    result = {}
    for name in ['combined-after','combined-dashboard','scene-only-dashboard','scene-only-soak-20260921','scene-only-final']:
        path = OUT / (name+'.json')
        if not path.exists():path=OUT/(name+'.checkpoint.json')
        if path.exists():
            result[name] = summarize(json.loads(path.read_text()))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
