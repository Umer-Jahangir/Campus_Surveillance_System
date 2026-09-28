"""Synthetic decisions test semantics only, not model accuracy."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scene_model import SceneWindow

class Model:
    score=.9
    def predict(self,frames):return {'score':self.score,'scope':'scene'}

class ObservationTests(unittest.TestCase):
    def active_window(self):
        model=Model();w=SceneWindow(model)
        for t in (5,6):w.update_clip({'frames':[None],'start':t-5,'end':t},t)
        self.assertTrue(w.active)
        return model,w
    def test_negative_is_end_but_reconnect_is_unknown(self):
        model,w=self.active_window();model.score=.1
        _,event=w.update_clip({'frames':[None],'start':2,'end':7},7)
        self.assertEqual(event['phase'],'end')
        self.assertIn('no longer detected',event['reason'])
        _,w=self.active_window()
        _,event=w.update_clip(None,0,epoch=1)
        self.assertEqual(event['phase'],'observation_lost')
        self.assertFalse(w.active)
        _,w=self.active_window()
        # A reconnect may already carry a complete window. Its first positive
        # decision must not swallow the interruption of the previous event.
        _,event=w.update_clip({'frames':[None],'start':0,'end':5},5,epoch=1)
        self.assertEqual(event['phase'],'observation_lost')
    def test_one_positive_decision_does_not_alert(self):
        model=Model();w=SceneWindow(model)
        _,e=w.update_clip({'frames':[None],'start':0,'end':5},5)
        self.assertIsNone(e)
        model.score=.1
        _,e=w.update_clip({'frames':[None],'start':1,'end':6},6)
        self.assertIsNone(e)
        self.assertFalse(w.active)

if __name__=='__main__':unittest.main()
