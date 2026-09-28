import os,sys,unittest,tempfile,json,threading,queue,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'Ultralytics')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,cv2,supervision as sv
from pipeline_frames import prepare_frame,original_boxes,annotated_jpeg
from fight_detector_onnx import FightDetectorONNX
from worker import detect_fight_groups,decoder_worker

class RegressionTests(unittest.TestCase):
    def test_original_coordinate_roundtrip_portrait_landscape_odd_sizes(self):
        for h,w in [(1080,1920),(1920,1080),(333,517)]:
            frame,g=prepare_frame(np.zeros((h,w,3),np.uint8),(256,320,3))
            expected=np.array([[w*.1,h*.2,w*.8,h*.9]],np.float32)
            boxes=expected.copy(); boxes[:,[0,2]]=boxes[:,[0,2]]*g['scale_x']+g['pad_x']; boxes[:,[1,3]]=boxes[:,[1,3]]*g['scale_y']+g['pad_y']
            np.testing.assert_allclose(original_boxes(boxes,g),expected,atol=.001)
            self.assertEqual(frame.shape,(256,320,3))

    def test_annotation_owns_inferred_pixels_after_source_changes(self):
        frame=np.zeros((256,320,3),np.uint8); frame[50:100,40:80]=255
        jpeg=annotated_jpeg(frame,[[40,50,80,100]],[7],{})
        frame[:]=0
        decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        self.assertGreater(decoded[70,60].mean(),240)
        self.assertLess(decoded[160,180].mean(),10)
        # Image and annotation share the same transform at every display size.
        for w,h in [(640,512),(160,128),(1280,1024)]:
            resized=cv2.resize(decoded,(w,h))
            self.assertGreater(resized[round(70*h/256),round(60*w/320)].mean(),235)

    def test_nearby_bystander_does_not_inherit_label(self):
        class Scores:
            def get_score(self,t): return {1:.95,2:.05}[t]
            def above_threshold(self,t): return self.get_score(t)>=.7
        alerts=detect_fight_groups([1,2],np.array([[0,0,20,80],[18,0,38,80]]),Scores())
        self.assertEqual(alerts[0]['track_ids'],[1]); self.assertEqual(alerts[0]['nearby_track_ids'],[2])

    def scorer(self):
        return FightDetectorONNX(str(ROOT/'src/models/lstm-violence-detection.onnx'))

    def test_window_gap_rewind_invalid_pose_and_empty_scene_clear_scores(self):
        fd=self.scorer(); pose=np.zeros((17,2),np.float32)
        for i in range(20): fd.update(1,pose,i/30)
        self.assertIn(1,fd.scores)
        fd.update(1,pose,10); self.assertNotIn(1,fd.scores); self.assertEqual(len(fd.buffers[1]),1)
        fd.update(1,pose,1); self.assertEqual(len(fd.buffers[1]),1)
        fd.update(1,np.full((17,2),np.nan),1.1); self.assertNotIn(1,fd.buffers)
        fd.update(2,pose,1); fd.cleanup([]); self.assertFalse(fd.buffers)

    def test_unverified_model_cannot_claim_fight(self):
        fd=self.scorer(); fd.scores[1]=.99
        self.assertFalse(fd.above_threshold(1))
        self.assertFalse(fd.diagnostics()["verified"])
        self.assertEqual(fd.diagnostics()["sequence_length"], 20)

    def test_missing_model_is_explicit(self):
        with self.assertRaisesRegex(FileNotFoundError,'FIGHT_MODEL_PATH'):
            FightDetectorONNX('does-not-exist.onnx')

    def test_keypoints_follow_tracker_filter_and_reordering(self):
        tracker=sv.ByteTrack()
        d=sv.Detections(xyxy=np.array([[10,10,40,90],[100,10,140,90]],np.float32),confidence=np.array([.9,.9]),class_id=np.array([0,0]),data={'pose':np.stack([np.ones((17,3)),np.full((17,3),2)])})
        first=tracker.update_with_detections(d)
        d2=d[np.array([1,0])]; second=tracker.update_with_detections(d2)
        for box,pose in zip(second.xyxy,second.data['pose']): self.assertEqual(pose[0,0],1 if box[0]<50 else 2)
        for _ in range(40): tracker.update_with_detections(sv.Detections.empty())
        self.assertFalse(tracker.tracked_tracks)

    def test_decoder_immutable_paced_timestamped_packets(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as td:
            path=str(Path(td)/'fixture.avi'); writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
            for i in range(5): writer.write(np.full((60,80,3),i*40,np.uint8))
            writer.release(); q=queue.Queue(8); stop=threading.Event(); ready=threading.Event(); ready.set()
            start=time.monotonic()
            with patch('worker.assign_cores_hybrid',return_value=[]):
                decoder_worker(0,path,(256,320,3),q,stop,ready,1,1)
            packets=[q.get_nowait() for _ in range(5)]
            self.assertGreaterEqual(time.monotonic()-start,.35)
            self.assertEqual([p['frame_id'] for p in packets],list(range(5)))
            np.testing.assert_allclose([p['source_time'] for p in packets],[i/10 for i in range(5)])
            self.assertLess(packets[0]['frame'].mean(),1)
            self.assertGreater(packets[-1]['frame'].mean(),100)
            self.assertFalse(np.shares_memory(packets[0]['frame'],packets[-1]['frame']))

if __name__=='__main__': unittest.main()
