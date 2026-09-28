"""Local RTSP publishers -> real decoders -> both models -> Socket.IO load test.

Publishers copy pre-encoded H.264 packets; no inference fixtures or accuracy
labels. Decode-to-Socket.IO age is NOT camera capture-to-screen latency.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'audit/validation-20260918'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'Ultralytics')

def run(count,seconds):
    import main,psutil
    config=OUT/'mediamtx-validation.yml'
    config.write_text('logLevel: warn\nrtspAddress: 127.0.0.1:18554\nrtspTransports: [tcp]\nrtmp: false\nhls: false\nwebrtc: false\nsrt: false\npaths:\n  all_others:\n')
    owned=[];handles=[];samples=[];frames=[];events=[];cpu_prev={};last=0;start=None
    client=main.app.test_client();sock=main.socketio.test_client(main.app)
    def launch(args,name):
        log=open(OUT/name,'w');handles.append(log)
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        owned.append(p);return p
    try:
        launch(['C:/mediamtx.exe',str(config)],'mediamtx-validation.log');time.sleep(1)
        urls=[]
        for i in range(count):
            url=f'rtsp://127.0.0.1:18554/validation{i}';urls.append(url)
            source=OUT/'720p15'/(['FF','NF','SF'][i%3]+'.mp4')
            launch(['C:/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe','-nostdin','-loglevel','error','-re','-stream_loop','-1','-i',str(source),'-an','-c:v','copy','-f','rtsp','-rtsp_transport','tcp',url],f'rtsp-publisher-{i}.log')
        time.sleep(2)
        if any(p.poll() is not None for p in owned):raise RuntimeError('Publisher/server failed; inspect owned-process logs')
        main.state['streams']=[{'url':u,'label':f'Validation camera {i+1}'} for i,u in enumerate(urls)]
        response=client.post('/api/start')
        if response.status_code!=200:raise RuntimeError(response.get_json())
        if not main.state['inference_alive'].wait(120):raise RuntimeError('Models not ready')
        start=time.monotonic();process=psutil.Process();psutil.cpu_percent()
        while time.monotonic()-start<seconds:
            elapsed=time.monotonic()-start
            for e in sock.get_received():
                if e['name']=='scene_alert':events.append({'wall_s':elapsed,**e['args'][0]})
                if e['name']!='frame_meta':continue
                m=e['args'][0];action=m.get('scene_action',{})
                if action.get('state')=='unavailable':raise RuntimeError(action.get('reason'))
                frames.append({'wall_s':elapsed,'stream':m['stream_id'],'frame_id':m['frame_id'],
                    'decode_to_socket_ms':(time.monotonic()-m['source_time'])*1000,
                    'window_age_ms':(time.monotonic()-action['window_end'])*1000 if action.get('window_end') is not None else None,
                    'queue_ms':m.get('queue_ms'),'pipeline_ms':m.get('pipeline_ms'),
                    'state':action.get('state'),'stats':m['stats']})
            if elapsed-last>=1:
                rss=0;cpu=0
                for p in [process]+process.children(recursive=True):
                    try:
                        rss+=p.memory_info().rss;c=p.cpu_times();total=c.user+c.system
                        if p.pid in cpu_prev:cpu+=(total-cpu_prev[p.pid])/(elapsed-last)*100
                        cpu_prev[p.pid]=total
                    except psutil.Error:pass
                samples.append({'wall_s':elapsed,'rss_mib':rss/2**20,'app_cpu_one_core_percent':cpu,
                    'host_cpu_percent':psutil.cpu_percent(),'queue_depth':[q.qsize() for q in main.state['meta_queues']],
                    'available_memory_mib':psutil.virtual_memory().available/2**20})
                last=elapsed
            time.sleep(.05)
        stats=client.get('/api/stats').get_json()
        # Deliberate publisher disconnect; verify observation loss separately
        # from a negative model decision. This tail is outside timed load run.
        for p in owned[1:]:p.terminate();p.wait(timeout=10)
        tail=time.monotonic()
        while time.monotonic()-tail<8:
            for e in sock.get_received():
                if e['name']=='scene_alert':events.append({'wall_s':time.monotonic()-start,'disconnect_test':True,**e['args'][0]})
            time.sleep(.1)
        data={'streams':count,'seconds':seconds,'profile':'720p15 RTSP TCP localhost',
            'method':'production Flask/Socket.IO, real RTSP decoders, pose and scene enabled; publisher copy overhead included; no browser paint/capture timestamp',
            'stats':stats,'samples':samples,'frames':frames,'events':events}
        path=OUT/f'rtsp_720p15_{count}.json';path.write_text(json.dumps(data,indent=2))
        print('SAVED',path,flush=True)
    finally:
        with main._lock:main._stop_pipeline()
        sock.disconnect()
        for p in reversed(owned):
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--streams',type=int,default=4);p.add_argument('--seconds',type=int,default=120);a=p.parse_args();run(a.streams,a.seconds)
