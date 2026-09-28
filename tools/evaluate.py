"""Replay labelled local videos through the same workers used by the dashboard."""
import argparse,json,multiprocessing as mp,os,platform,queue,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,psutil
from evaluation import validate_manifest,metrics
from fight_detector_onnx import FightDetectorONNX
from worker import decoder_worker,inference_worker


def evaluate(clip,root):
    path=str((root/clip['path']).resolve()); cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError(f'Cannot open {path}')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if count<=0: raise ValueError('Evaluation requires a local file with a known frame count')
    stop=mp.Event(); ready=mp.Event(); mq=mp.Queue(8); rq=mp.Queue(8)
    inf=mp.Process(target=inference_worker,args=(1,(256,320,3),[mq],rq,stop,ready,1,1))
    dec=mp.Process(target=decoder_worker,args=(0,path,(256,320,3),mq,stop,ready,1,1))
    records=[]; started=time.perf_counter(); inf.start(); dec.start()
    try:
        while True:
            try: result=rq.get(timeout=60)
            except queue.Empty: raise RuntimeError('No result for 60 seconds; evaluation incomplete')
            if result['type']=='error': raise RuntimeError(result['error'])
            records.append(result)
            if result['frame_id']==count-1: break
    finally:
        stop.set()
        for process in (dec,inf):
            process.join(5)
            if process.is_alive(): process.terminate(); process.join(3)
        for q in (mq,rq): q.cancel_join_thread(); q.close()
    elapsed=time.perf_counter()-started
    if [r['frame_id'] for r in records]!=list(range(count)):
        raise RuntimeError('Evaluation dropped frames; metrics would be incomplete')
    if clip['fight_intervals'] and clip['fight_intervals'][-1][1]>records[-1]['source_time']+1:
        raise ValueError('Labels extend beyond video duration')
    samples=[(r['source_time'],bool(r['behavior_alerts'])) for r in records]
    return dict(path=clip['path'],frames=count,metrics=metrics(samples,clip['fight_intervals']),
                wall_seconds_including_startup=elapsed,
                model_mean_ms=float(np.mean([r['model_lat'] for r in records])),
                model_p95_ms=float(np.percentile([r['model_lat'] for r in records],95)),
                predictions=[dict(time=t,fight=p) for t,p in samples])


def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest',type=Path); p.add_argument('--split',choices=['tune','evaluation'],default='evaluation'); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    clips=json.loads(args.manifest.read_text())['clips']; validate_manifest(clips,args.manifest.parent)
    fd=FightDetectorONNX(os.environ.get('FIGHT_MODEL_PATH',str(ROOT/'src/models/lstm-violence-detection.onnx')))
    if not fd.verified: raise ValueError('Accuracy evaluation requires a verified FIGHT_CONTRACT; raw scores do not establish fight labels')
    selected=[c for c in clips if c['split']==args.split]
    if not selected: raise ValueError('No clips in selected split')
    results=[evaluate(c,args.manifest.parent) for c in selected]
    report=dict(split=args.split,threshold=fd.threshold,model=os.environ.get('FIGHT_MODEL_PATH','bundled LSTM'),contract=fd.contract,
                hardware=platform.processor(),physical_cpus=psutil.cpu_count(logical=False),providers=['CPUExecutionProvider'],clips=results)
    args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__': mp.freeze_support(); main()
