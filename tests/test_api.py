import os,sys,unittest,threading,hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'Ultralytics'); sys.path.insert(0,str(ROOT/'src'))
import main
from fight_detector_onnx import FightDetectorONNX

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.saved=main.state.copy(); main.state.update(running=False,restarting=False,streams=[],latest_meta={},stats={},inference_proc=None)
        self.client=main.app.test_client()
    def tearDown(self): main.state.clear(); main.state.update(self.saved)
    def test_registration_validation_and_restart_guard(self):
        self.assertEqual(self.client.post('/api/start').status_code,400)
        self.assertEqual(self.client.post('/api/streams',json={}).status_code,400)
        self.assertEqual(self.client.post('/api/streams',json={'url':'missing.avi'}).status_code,201)
        self.assertEqual(self.client.post('/api/start').status_code,400)
        main.state['restarting']=True
        self.assertEqual(self.client.post('/api/streams',json={'url':'x'}).status_code,409)
        self.assertEqual(self.client.delete('/api/streams/0').status_code,409)
        self.assertEqual(self.client.post('/api/stop').status_code,409)
    def test_new_viewer_does_not_reset_existing_viewer_and_receives_cache(self):
        first=main.socketio.test_client(main.app); first.get_received()
        main.state['latest_meta']={0:{'stream_id':0,'frame_id':9,'published_at':1}}
        second=main.socketio.test_client(main.app)
        self.assertEqual(first.get_received(),[])
        received=second.get_received()
        self.assertEqual([e['name'] for e in received],['initial_state','frame_meta'])
        self.assertEqual(received[-1]['args'][0]['frame_id'],9)
        first.disconnect(); second.disconnect()
    def test_two_mjpeg_viewers_get_same_existing_frame_and_stop_on_generation_change(self):
        main.state.update(running=True,stop_event=threading.Event())
        with main._mjpeg_lock: main._mjpeg_frames[0]=b'test-jpeg'
        a=main._mjpeg_generator(0); b=main._mjpeg_generator(0)
        chunk=next(a)
        self.assertEqual(chunk,next(b))
        self.assertTrue(chunk.endswith(main.MJPEG_BOUNDARY+b"\r\n"))
        with patch.object(main.time, 'monotonic', return_value=10**12):
            repeat=next(a)
        self.assertIn(b'test-jpeg',repeat)
        self.assertTrue(repeat.endswith(main.MJPEG_BOUNDARY+b"\r\n"))
        main.state['stop_event']=threading.Event()
        with self.assertRaises(StopIteration): next(a)
        with self.assertRaises(StopIteration): next(b)
        main._mjpeg_frames.clear()
    def test_verified_contract_is_bound_to_weights_and_sampling(self):
        model=ROOT/'src/models/lstm-violence-detection.onnx'
        contract=dict(verified=True,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),keypoints='coco17_xy',scope='person',positive_label='fight',coordinates='inference_pixels',sample_fps=10)
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as td:
            p=Path(td)/'contract.json'; p.write_text(json.dumps(contract))
            with patch.dict(os.environ,FIGHT_CONTRACT=str(p)):
                fd=FightDetectorONNX(model); self.assertTrue(fd.verified)
                import numpy as np
                for i in range(20): fd.update(1,np.zeros((17,2)),i/10)
                self.assertIn(1,fd.scores)
                fd.update(1,np.zeros((17,2)),2.2); self.assertNotIn(1,fd.scores)
                contract['model_sha256']='wrong'; p.write_text(json.dumps(contract))
                with self.assertRaisesRegex(ValueError,'sha256'): FightDetectorONNX(model)

if __name__=='__main__': unittest.main()
