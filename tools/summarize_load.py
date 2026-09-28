"""Summarize saved measurements without rerunning inference."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/validation-20260918'

def fmt(x):return '—' if x is None else f'{x:.2f}'

lines=['# Measured camera-load results','',
    'Intel Core i5-1235U, 10 physical / 12 logical CPUs. CPU inference only. Each completed case runs 120 seconds; throughput/latency exclude the first 15 seconds. No annotation-based accuracy is inferred. Separate run conditions and changing host load can affect comparisons.','',
    '| Input | Cameras | FPS per camera | P95 receipt latency ms per camera | P95 queue ms per camera | Max decision gap s per camera | Mean application CPU (one core = 100%) | Peak process-tree RSS MiB |',
    '|---|---:|---|---|---|---|---:|---:|']
details=[]
for profile in ['current','720p15']:
    for n in [1,2,4]:
        p=OUT/f'load_{profile}_{n}.json'
        if not p.exists():continue
        d=json.loads(p.read_text());ss=d['summary']
        values=[', '.join(fmt(s['processed_fps_after_15s']) for s in ss),
            ', '.join(fmt(s['receipt_ms']['p95'] if s['receipt_ms'] else None) for s in ss),
            ', '.join(fmt(s['queue_ms']['p95'] if s['queue_ms'] else None) for s in ss),
            ', '.join(fmt(s['decision_gap_seconds']['max'] if s['decision_gap_seconds'] else None) for s in ss)]
        lines.append(f'| {profile} | {n} | '+ ' | '.join(values)+f" | {fmt(d['cpu']['mean'])} | {fmt(d['rss_mib']['max'])} |")
        detail={'profile':profile,'cameras':n,'source':str(p),'expected_frames_per_camera':d['input_fps']*d['seconds'],
            'generated':[s['generated'] for s in ss],'received':[s['received'] for s in ss],
            'not_received_by_deadline':[s['not_received_by_deadline'] for s in ss],
            'max_queue_depth':[max(x['queue_depth'][i] for x in d['samples']) for i in range(n)],
            'rss_first_10_mean_mib':statistics.mean(x['rss_mib'] for x in d['samples'][:10]),
            'rss_last_10_mean_mib':statistics.mean(x['rss_mib'] for x in d['samples'][-10:]),
            'application_cpu_machine_fraction_percent':d['cpu']['mean']/d['logical_cores'],
            'host_cpu':d['host_cpu'],'producer_errors':d['producer_errors'],'streams':[]}
        for i in range(n):
            all_frames=[f for f in d['frames'] if f['stream']==i]
            last_id=max((f['frame_id'] for f in all_frames),default=-1)
            fs=[f for f in d['frames'] if f['stream']==i and f['wall_s']>=15]
            first={}
            for f in fs:
                if f['window_end'] is not None:first.setdefault(f['window_end'],f)
            ev=next((e for e in d['events'] if e['stream']==i and e['phase']=='start'),None)
            detail['streams'].append({'stream':i,'unique_decisions_after_15s':len(first),
                'skipped_before_last_received_frame':last_id+1-len(all_frames),
                'unreceived_tail_at_deadline':ss[i]['generated']-last_id-1,
                'action_ms_mean':statistics.mean(f['action_ms'] for f in first.values()) if first else None,
                'first_start_wall_s':ev['wall_s'] if ev else None,
                'first_start_window_end':ev['window_end'] if ev else None,
                'first_start_receipt_minus_window_end_s':ev['wall_s']-ev['window_end'] if ev else None,
                'receipt_first_half_ms':ss[i]['first_half_receipt_ms'],
                'receipt_second_half_ms':ss[i]['second_half_receipt_ms']})
        details.append(detail)
lines+=['','Receipt latency runs from scheduled virtual capture through production inference/JPEG/IPC. It excludes RTSP and browser paint. Queue depth is sampled; queues have hard capacity eight. “Not received” includes frame skipping and the in-flight tail at cutoff. A producer failing to generate at target FPS invalidates a claim that all target cameras were sustained, even when queue depth stays bounded.','',
    'Current-profile streams cycle FF (368×358), NF (1080×1920), SF (360×640), FF at 30 FPS. Target-profile sources are pre-encoded, aspect-contained 1280×720/15 FPS H.264 files. These are paced, repeated real clips, not independent camera recordings or accuracy data.','',
    'Detailed counts, sampled queue maxima, memory trend, host CPU and first-alert timing are in `load_summary.json`.']
(OUT/'MEASUREMENTS.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'load_summary.json').write_text(json.dumps(details,indent=2))
print('\n'.join(lines))
