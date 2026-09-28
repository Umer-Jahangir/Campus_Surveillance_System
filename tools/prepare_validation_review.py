"""Create neutral timestamped excerpts and inventory local footage, no model labels."""
from pathlib import Path
import cv2,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/validation-20260918'
SOURCE=Path.home()/'Downloads/Video'
FFMPEG=Path('C:/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe')

def excerpt(name,start,end):
    cap=cv2.VideoCapture(str(SOURCE/f'{name}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,round(start*fps));i=round(start*fps)
    temp=OUT/f'{name}_{start:g}_{end:g}_raw.avi';target=OUT/f'{name}_{start:g}_{end:g}_review.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'MJPG'),fps,(720,760))
    while i/fps<end:
        ok,frame=cap.read()
        if not ok:break
        import numpy as np
        canvas=np.zeros((760,720,3),np.uint8)
        h,w=frame.shape[:2];s=min(720/w,680/h);f=cv2.resize(frame,(round(w*s),round(h*s)))
        y=(680-f.shape[0])//2;x=(720-f.shape[1])//2;canvas[y:y+f.shape[0],x:x+f.shape[1]]=f
        cv2.putText(canvas,f'{name} | source {i/fps:.2f}s',(15,712),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.putText(canvas,'Unlabelled review - no model predictions',(15,747),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        writer.write(canvas);i+=1
    cap.release();writer.release()
    subprocess.run([str(FFMPEG),'-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    # Only remove the exact intermediate created by this function inside OUT.
    temp.unlink()
    return str(target)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    target=OUT/'720p15';target.mkdir(exist_ok=True)
    for name in ['FF','NF','SF']:
        if not (target/f'{name}.mp4').exists():
            subprocess.run([str(FFMPEG),'-y','-loglevel','error','-i',str(SOURCE/f'{name}.mp4'),
                '-vf','scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=15',
                '-an','-c:v','libx264','-preset','fast','-crf','20',str(target/f'{name}.mp4')],check=True)
    rows=[]
    for path in sorted(SOURCE.glob('*')):
        if path.suffix.lower() not in ('.mp4','.avi','.mkv','.mov'):continue
        cap=cv2.VideoCapture(str(path));fps=cap.get(5);frames=cap.get(7)
        rows.append({'path':str(path),'duration_seconds':frames/fps if fps else None,
            'fps':fps,'dimensions':[cap.get(3),cap.get(4)],'label_source':None,
            'training_overlap':'unknown','review_status':'provisional' if path.stem in ['SF','NF','FF'] else 'not reviewed; filename not used as ground truth'})
        cap.release()
    (OUT/'local_inventory.json').write_text(json.dumps(rows,indent=2))
    files=[excerpt('FF',0,6),excerpt('SF',0,8),excerpt('SF',7,15),excerpt('SF',14,20.8667),excerpt('NF',0,11.4667)]
    (OUT/'review_files.json').write_text(json.dumps(files,indent=2));print('\n'.join(files))
