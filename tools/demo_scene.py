"""Real model replay. No labels or scores are hardcoded from source filenames."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'Ultralytics'))
import cv2,numpy as np
from scene_model import SceneModel,SceneWindow,configured_scene_path

def main():
    model=SceneModel(configured_scene_path())
    out=ROOT/'audit/scene_demo';out.mkdir(exist_ok=True)
    for name in ['SF','NF','FF']:
        path=Path.home()/'Downloads/Video'/f'{name}.mp4'
        cap=cv2.VideoCapture(str(path));fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4))
        scale=min(1,720/max(h,w));size=(int(w*scale)//2*2,int(h*scale)//2*2)
        writer=cv2.VideoWriter(str(out/f'{name}_scene.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
        window=SceneWindow(model);records=[];events=[];i=0;started=time.perf_counter()
        while True:
            ok,frame=cap.read()
            if not ok:break
            status,event=window.update(cv2.resize(frame,(224,224)),i/fps)
            if event:events.append(event)
            if status.get('window_end')==i/fps:records.append(dict(status))
            display=cv2.resize(frame,size)
            cv2.rectangle(display,(0,0),(size[0],72),(15,15,15),-1)
            color=(40,60,255) if status['state']=='suspected_fight' else (230,240,240)
            lines=[f"{status['state']} | t={i/fps:.2f}s",'X3D-M scene | threshold .5 | confirm 2']
            if status.get('score') is not None:lines.append(f"p(violent)={status['score']:.3f} | {status['window_start']:.1f}-{status['window_end']:.1f}s")
            for j,line in enumerate(lines):cv2.putText(display,line,(4,18+20*j),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
            writer.write(display);i+=1
        cap.release();writer.release()
        if window.active:events.append({'phase':'observation_lost','reason':'end of clip; fight outcome unknown','source_time':i/fps})
        report={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'frames':i,'fps':fps,'elapsed_seconds':time.perf_counter()-started,
                'model_sha256':model.sha256,'predictions':records,'events':events,
                'annotations':'pending review; filenames not used as labels'}
        (out/f'{name}.json').write_text(json.dumps(report,indent=2))
        print(name, 'frames',i,'scores',[(round(x['window_end'],2),round(x['score'],3)) for x in records], 'events',events,flush=True)

if __name__=='__main__':main()
