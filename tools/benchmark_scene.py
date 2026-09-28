"""Paced virtual-camera load test of production pose + scene worker.

No accuracy evidence: loops unlabelled real footage, with continuous timestamps.
Excludes RTSP transport and browser paint. Includes decoder/resize, queue, models,
tracking, JPEG encoding and IPC receipt. Every case starts fresh worker state.
"""
import argparse,json,os,sys,time,threading,queue,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'Ultralytics')

def run_case(count,fps,size,seconds,out):
    import cv2,numpy as np,psutil
    from worker import inference_worker
    from scene_model import SceneSampler
    from pipeline_frames import prepare_frame
    for key,value in [('SCENE_WINDOW_SECONDS',5),('SCENE_STRIDE_SECONDS',1),('SCENE_THRESHOLD',.5),('SCENE_CONFIRM_WINDOWS',2)]:
        if float(os.environ.get(key,value))!=value:raise ValueError('Baseline benchmark requires default '+key)
    queues=[mp.Queue(8) for _ in range(count)];results=mp.Queue(8)
    stop=mp.Event();ready=mp.Event()
    cpp=max(1,8//count)
    worker=mp.Process(target=inference_worker,args=(count,(256,320,3),queues,results,stop,ready,count,cpp))
    worker.start()
    if not ready.wait(120):
        stop.set();worker.terminate();raise RuntimeError('Model startup failed')
    started=time.perf_counter(); generated=[0]*count;producer_errors=[]
    def produce(i):
        name=['FF','NF','SF'][i%3]+'.mp4'
        source=(ROOT/'audit/validation-20260918/720p15'/name) if size else (Path.home()/'Downloads/Video'/name)
        cap=cv2.VideoCapture(str(source))
        sampler=SceneSampler();frame_id=0
        try:
            while not stop.is_set() and time.perf_counter()-started<seconds:
                due=started+frame_id/fps
                if stop.wait(max(0,due-time.perf_counter())):break
                ok,frame=cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,frame=cap.read()
                if not ok:raise RuntimeError('Source decode failed')
                if size and frame.shape[:2]!=(size[1],size[0]):
                    raise RuntimeError('Expected pre-encoded 720p15 source')
                clip=sampler.update(cv2.resize(frame,(224,224)),frame_id/fps,0)
                frame,geometry=prepare_frame(frame,(256,320,3))
                packet={'stream_id':i,'frame':frame,'timestamp':due,'frame_id':frame_id,
                        'source_time':frame_id/fps,'epoch':0,'geometry':geometry,
                        'scene_clip':clip,'source_fps':fps,'is_file':False}
                try:
                    if queues[i].full():queues[i].get_nowait()
                    queues[i].put_nowait(packet)
                except queue.Empty:pass
                except queue.Full:pass
                frame_id+=1;generated[i]=frame_id
        except Exception as e:producer_errors.append(str(e))
        finally:cap.release()
    threads=[threading.Thread(target=produce,args=(i,),daemon=True) for i in range(count)]
    for t in threads:t.start()
    proc=psutil.Process();cpu_prev={};samples=[];frames=[];events=[];last_sample=0
    psutil.cpu_percent()
    try:
        while time.perf_counter()-started<seconds:
            try:
                msg=results.get(timeout=.1)
                if msg['type']=='error':raise RuntimeError(msg)
                elapsed=time.perf_counter()-started
                action=msg['scene_action']
                if action['state']=='unavailable':raise RuntimeError(action.get('reason','Scene model unavailable'))
                frames.append({'wall_s':elapsed,'stream':msg['stream_id'],'frame_id':msg['frame_id'],
                    'source_s':msg['source_time'],'queue_ms':msg['queue_ms'],
                    'receipt_ms':(time.perf_counter()-msg['timestamp'])*1000,
                    'pipeline_ms':msg['e2e_lat'],'pose_ms':msg['model_lat'],
                    'window_end':action.get('window_end'),'action_ms':action.get('action_ms'),
                    'state':action['state']})
                if msg.get('scene_event'):events.append({'wall_s':elapsed,'stream':msg['stream_id'],**msg['scene_event']})
            except queue.Empty:pass
            elapsed=time.perf_counter()-started
            if elapsed-last_sample>=1:
                rss=0;cpu=0
                for p in [proc]+proc.children(recursive=True):
                    try:
                        rss+=p.memory_info().rss;c=p.cpu_times();total=c.user+c.system
                        if p.pid in cpu_prev:cpu+=max(0,total-cpu_prev[p.pid])/(elapsed-last_sample)*100
                        cpu_prev[p.pid]=total
                    except psutil.Error:pass
                samples.append({'wall_s':elapsed,'rss_mib':rss/2**20,'app_cpu_one_core_percent':cpu,
                    'host_cpu_percent':psutil.cpu_percent(),'available_memory_mib':psutil.virtual_memory().available/2**20,
                    'queue_depth':[q.qsize() for q in queues]})
                last_sample=elapsed
    finally:
        stop.set()
        for t in threads:t.join(5)
        worker.join(10)
        if worker.is_alive():worker.terminate();worker.join()
        for q in queues+[results]:q.cancel_join_thread();q.close()
    def stats(values):
        return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'max':float(max(values))} if values else None
    measured=[f for f in frames if f['wall_s']>=15]
    summary=[]
    for i in range(count):
        fs=[f for f in measured if f['stream']==i]
        decisions={f['window_end']:f for f in fs if f['window_end'] is not None}
        ends=sorted(decisions)
        summary.append({'stream':i,'processed_fps_after_15s':len(fs)/(seconds-15),
            'generated':generated[i],'received':sum(f['stream']==i for f in frames),
            'not_received_by_deadline':generated[i]-sum(f['stream']==i for f in frames),
            'queue_ms':stats([f['queue_ms'] for f in fs]),'receipt_ms':stats([f['receipt_ms'] for f in fs]),
            'action_window_age_ms':stats([(f['wall_s']-f['window_end'])*1000 for f in fs if f['window_end'] is not None]),
            'decision_gap_seconds':stats([b-a for a,b in zip(ends,ends[1:])]),
            'first_half_receipt_ms':stats([f['receipt_ms'] for f in fs if f['wall_s']<seconds/2]),
            'second_half_receipt_ms':stats([f['receipt_ms'] for f in fs if f['wall_s']>=seconds/2])})
    data={'method':'paced local virtual cameras; repeated footage, not accuracy evaluation; no RTSP/browser',
        'streams':count,'input_fps':fps,'input_size':size or 'original','seconds':seconds,
        'gpu_inference':False,'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),
        'model_settings':{'window_seconds':5,'stride_seconds':1,'threshold':.5,'confirm_windows':2},
        'summary':summary,'cpu':stats([s['app_cpu_one_core_percent'] for s in samples[1:]]),
        'host_cpu':stats([s['host_cpu_percent'] for s in samples[1:]]),
        'rss_mib':stats([s['rss_mib'] for s in samples]),'samples':samples,'frames':frames,'events':events,
        'producer_errors':producer_errors}
    out.write_text(json.dumps(data,indent=2));print(json.dumps({'output':str(out),'summary':summary,'cpu':data['cpu'],'rss':data['rss_mib']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=120);p.add_argument('--streams',type=int,nargs='+',default=[1,2,4]);p.add_argument('--profile',choices=['current','720p15','both'],default='both');a=p.parse_args()
    out=ROOT/'audit/validation-20260918';out.mkdir(exist_ok=True)
    for profile,fps,size in [('current',30,None),('720p15',15,(1280,720))]:
        if a.profile not in (profile,'both'):continue
        for n in a.streams:run_case(n,fps,size,a.seconds,out/f'load_{profile}_{n}.json')
