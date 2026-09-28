"""One-camera full-app comparison/soak with the real dashboard available.

Same pre-encoded FF 720p15 source as the previous full-app baseline. Samples
record stage boundaries, actual decoder FPS, decision cadence and memory.
Browser verification is performed separately through the visible dashboard.
"""
import argparse,json,os,sys,time,subprocess,threading,queue,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'audit/validation-20260918/focused'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'Ultralytics')

def run(args):
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    os.environ['DETECTION_MODE']=args.mode
    import main,psutil
    OUT.mkdir(exist_ok=True)
    target=OUT/(args.name+'.json')
    if target.exists():raise ValueError('Preserve existing result; choose a new --name')
    config=OUT/'mediamtx.yml'
    config.write_text('logLevel: warn\nrtspAddress: 127.0.0.1:18554\nrtspTransports: [tcp]\nrtmp: false\nhls: false\nwebrtc: false\nsrt: false\npaths:\n  all_others:\n')
    owned=[];logs=[];samples=[];frames=[];events=[];failure=None;started=None
    def launch(command,label):
        f=open(OUT/f'{args.name}-{label}.log','w');logs.append(f)
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);owned.append(p)
        return p
    import socketio
    client=main.app.test_client();sock=socketio.Client(reconnection=False)
    incoming=queue.Queue()
    sock.on('frame_meta',lambda data:incoming.put({'name':'frame_meta','args':[data]}))
    sock.on('scene_alert',lambda data:incoming.put({'name':'scene_alert','args':[data]}))
    def received():
        result=[]
        while True:
            try:result.append(incoming.get_nowait())
            except queue.Empty:return result
    try:
        launch(['C:/mediamtx.exe',str(config)],'server');time.sleep(1)
        source=ROOT/'audit/validation-20260918/720p15/FF.mp4'
        url='rtsp://127.0.0.1:18554/focused'
        launch(['C:/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe','-nostdin','-loglevel','error','-re','-stream_loop','-1','-i',str(source),'-an','-c:v','copy','-f','rtsp','-rtsp_transport','tcp',url],'publisher')
        time.sleep(2)
        if any(p.poll() is not None for p in owned):raise RuntimeError('RTSP startup failed')
        main.state['streams']=[{'url':url,'label':'One camera · '+args.mode}]
        threading.Thread(target=lambda:main.socketio.run(main.app,host='127.0.0.1',port=5002,use_reloader=False,allow_unsafe_werkzeug=True),daemon=True).start()
        for attempt in range(30):
            try:sock.connect('http://127.0.0.1:5002',transports=['polling']);break
            except Exception:
                if attempt==29:raise
                time.sleep(.1)
        result=client.post('/api/start')
        if result.status_code!=200:raise RuntimeError(result.get_json())
        if not main.state['inference_alive'].wait(120):raise RuntimeError('Inference startup failed')
        print('READY http://127.0.0.1:5002 mode='+args.mode,flush=True)
        started=time.monotonic();last=0;cpu_prev={};process=psutil.Process();low=0
        psutil.cpu_percent()
        while time.monotonic()-started<args.seconds:
            elapsed=time.monotonic()-started
            for event in received():
                if event['name']=='scene_alert':events.append({'elapsed':elapsed,**event['args'][0]})
                if event['name']!='frame_meta':continue
                m=event['args'][0]
                if m['scene_action']['state']=='unavailable':raise RuntimeError(m['scene_action'].get('reason'))
                frames.append({'elapsed':elapsed,'frame_id':m['frame_id'],'source_time':m['source_time'],
                    'receipt_ms':(time.monotonic()-m['source_time'])*1000,
                    'queue_ms':m.get('queue_ms'),'delivery_ms':m.get('delivery_ms'),
                    'pipeline_ms':m.get('pipeline_ms'),'profile':m.get('profile'),
                    'action':m['scene_action'],'stats':m['stats']})
            if elapsed-last>=1:
                memory=psutil.virtual_memory();rss=0;cpu=0;processes=[]
                for p in [process]+process.children(recursive=True):
                    try:
                        resident=p.memory_info().rss;rss+=resident;c=p.cpu_times();total=c.user+c.system
                        if p.pid in cpu_prev:cpu+=(total-cpu_prev[p.pid])/(elapsed-last)*100
                        cpu_prev[p.pid]=total
                        processes.append({'pid':p.pid,'name':p.name(),'rss_mib':resident/2**20,'threads':p.num_threads()})
                    except psutil.Error:pass
                samples.append({'elapsed':elapsed,'rss_mib':rss/2**20,'available_mib':memory.available/2**20,
                    'host_cpu_percent':psutil.cpu_percent(),'app_cpu_one_core_percent':cpu,
                    'frame_queues':[q.qsize() for q in main.state['meta_queues']],
                    'scene_queues':[q.qsize() for q in main.state['scene_queues']], 'processes':processes})
                low=low+1 if memory.available<256*2**20 else 0
                if low>=3:raise RuntimeError('Stopped early: host available memory <256 MiB for 3 samples')
                if main.state.get('error'):raise RuntimeError(main.state['error'])
                if int(elapsed)//60>int(last)//60:
                    checkpoint=target.with_suffix('.checkpoint.json')
                    temporary=checkpoint.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'mode':args.mode,'requested_seconds':args.seconds,
                        'observed_seconds':elapsed,'failure':'incomplete checkpoint; not a completed soak',
                        'samples':samples,'frames':frames,'events':events}))
                    temporary.replace(checkpoint)
                    target.with_suffix('.progress.json').write_text(json.dumps({'elapsed':elapsed,'last':frames[-1] if frames else None,'memory':samples[-1]},indent=2))
                    print(f'PROGRESS {elapsed:.1f}s RSS={rss/2**20:.1f} MiB available={memory.available/2**20:.1f} MiB',flush=True)
                last=elapsed
            time.sleep(.05)
    except Exception as e:
        failure=str(e);print('FAILURE',failure,flush=True)
    finally:
        duration=time.monotonic()-started if started is not None else 0
        data={'mode':args.mode,'requested_seconds':args.seconds,'observed_seconds':duration,'failure':failure,
            'input':'same FF-derived 1280x720/15 FPS H264 RTSP TCP baseline source',
            'boundaries':'receipt = decoder frame return through inference/JPEG/IPC/Socket.IO; excludes pre-decode buffering and browser paint',
            'samples':samples,'frames':frames,'events':events}
        target.write_text(json.dumps(data,indent=2));print('SAVED',target,flush=True)
        with main._lock:main._stop_pipeline()
        sock.disconnect()
        for p in reversed(owned):
            if p.poll() is None:p.terminate();p.wait(timeout=10)
        for f in logs:f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['combined','scene_only'],required=True);p.add_argument('--seconds',type=int,default=120);p.add_argument('--name',required=True);run(p.parse_args())
