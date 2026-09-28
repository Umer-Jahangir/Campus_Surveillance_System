"""Timestamped half-second contact sheets for proposed annotation review."""
from pathlib import Path
import cv2,numpy as np
out=Path(__file__).resolve().parents[1]/'audit/scene_demo';out.mkdir(exist_ok=True)
for name in ['SF','NF','FF']:
    cap=cv2.VideoCapture(str(Path.home()/'Downloads/Video'/f'{name}.mp4'))
    fps=cap.get(5);tiles=[];i=0
    while True:
        ok,f=cap.read()
        if not ok:break
        if i%round(fps/2)==0:
            h,w=f.shape[:2];scale=min(180/w,180/h)
            small=cv2.resize(f,(round(w*scale),round(h*scale)))
            tile=np.zeros((204,180,3),np.uint8);tile[:small.shape[0],:small.shape[1]]=small
            cv2.putText(tile,f'{i/fps:.2f}s',(4,198),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(tile)
        i+=1
    cap.release()
    for start in range(0,len(tiles),20):
        sheet=np.zeros((4*204,5*180,3),np.uint8)
        for j,t in enumerate(tiles[start:start+20]):sheet[j//5*204:(j//5+1)*204,j%5*180:(j%5+1)*180]=t
        cv2.imwrite(str(out/f'{name}_review_{start//20}.jpg'),sheet)
