import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scene_model import SceneWindow,SceneModel,SceneSampler,preprocess_scene

class SceneTests(unittest.TestCase):
    def test_probability_head_is_not_softmaxed_twice(self):
        import torch
        m=SceneModel.__new__(SceneModel)
        m.model=lambda tensor:torch.tensor([[.1,.9]])
        result=m.predict([np.zeros((2,2,3),np.uint8)])
        self.assertAlmostEqual(result['score'],.9,places=6)
    def test_rgb_normalization_and_temporal_order(self):
        a=np.zeros((8,9,3),np.uint8);a[:,:,2]=255
        b=np.zeros_like(a);b[:,:,0]=255
        x=preprocess_scene([a,b])
        self.assertEqual(x.shape,(1,3,16,224,224))
        self.assertAlmostEqual(float(x[0,0,0,0,0]),(1-.45)/.225,places=5)
        self.assertAlmostEqual(float(x[0,2,-1,0,0]),(1-.45)/.225,places=5)
    def test_fixture_event_start_end_and_discontinuity(self):
        class Fixture:
            score=.9
            def predict(self,frames):return {'score':self.score,'scope':'scene'}
        m=Fixture();w=SceneWindow(m);f=np.zeros((2,2,3),np.uint8);events=[]
        for i in range(71):
            s,e=w.update(f,i/10)
            if e:events.append(e)
        self.assertEqual([e['phase'] for e in events],['start'])
        m.score=.1;s,e=w.update(f,8)
        self.assertEqual(e['phase'],'end');self.assertEqual(s['state'],'active')
        s,e=w.update(f,0,1);self.assertEqual(s['state'],'warming_up');self.assertIsNone(s['score'])
        self.assertNotIn('track_ids',s)
    def test_unavailable_has_reason(self):
        s,e=SceneWindow(None,'weights missing').update(None,0)
        self.assertEqual(s['state'],'unavailable');self.assertEqual(s['reason'],'weights missing')

    def test_live_skipped_pose_frames_keep_complete_action_windows(self):
        class Fixture:
            def predict(self,frames):
                self.frames=frames
                return {'score':.9,'scope':'scene'}
        sampler=SceneSampler();model=Fixture();window=SceneWindow(model)
        clips=[]
        for i in range(211):
            clip=sampler.update(np.full((2,2,3),i,np.uint8),i/30,0)
            if i in (149,210):clips.append((clip,i/30))
        first=clips[0][0]['frames'].copy()
        a,e=window.update_clip(*clips[0],epoch=0)
        a,e=window.update_clip(*clips[1],epoch=0)
        self.assertEqual(e['phase'],'start')
        self.assertEqual(model.frames.shape,(16,2,2,3))
        self.assertTrue(np.array_equal(first,clips[0][0]['frames']))
        a,e=window.update_clip(*clips[1],epoch=0);self.assertIsNone(e)
        a,e=window.update_clip(None,9,epoch=0)
        self.assertEqual(a['state'],'warming_up');self.assertEqual(e['phase'],'observation_lost')
        self.assertIsNone(sampler.update(np.zeros((2,2,3),np.uint8),0,1))

if __name__=='__main__':unittest.main()
