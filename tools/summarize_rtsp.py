"""Reproducible summary of full-application RTSP measurements."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'audit/validation-20260918'
reports=[]
for path in sorted(OUT.glob('rtsp_720p15_[124].json')):
    d=json.loads(path.read_text());samples=d['samples']
    r={'source':str(path),'cameras':d['streams'],'seconds':d['seconds'],'streams':[],
       'mean_app_cpu_one_core_percent':float(np.mean([s['app_cpu_one_core_percent'] for s in samples[1:]])),
       'mean_host_cpu_percent':float(np.mean([s['host_cpu_percent'] for s in samples[1:]])),
       'peak_rss_mib':max(s['rss_mib'] for s in samples),
       'min_host_available_memory_mib':min(s['available_memory_mib'] for s in samples),
       'event_counts':{},'disconnect_events':[e for e in d['events'] if e.get('disconnect_test')]}
    for event in d['events']:r['event_counts'][event['phase']]=r['event_counts'].get(event['phase'],0)+1
    for i in range(d['streams']):
        frames=[f for f in d['frames'] if f['stream']==i and f['wall_s']>=15]
        before=[f for f in d['frames'] if f['stream']==i and f['wall_s']<15]
        stats=d['stats']['stats'][str(i)]
        p95=lambda key:float(np.percentile([f[key] for f in frames if f[key] is not None],95)) if frames else None
        r['streams'].append({'stream':i,'frames':stats['frames'],'skipped_before_last':stats['dropped'],
            'processed_fps_after_15s':(stats['frames']-(before[-1]['stats']['frames'] if before else 0))/(d['seconds']-15),
            'p95_decode_to_socket_ms':p95('decode_to_socket_ms'),'p95_queue_ms':p95('queue_ms'),
            'p95_window_age_ms':p95('window_age_ms'),
            'max_queue_depth':max(s['queue_depth'][i] for s in samples)})
    reports.append(r)
(OUT/'rtsp_summaries.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
