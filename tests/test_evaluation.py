import sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from evaluation import metrics,validate_manifest

class EvaluationTests(unittest.TestCase):
    def test_metrics_delay_and_undefined_precision(self):
        result=metrics([(0,True),(1,False),(2,True),(3,False)],[(1,3)])
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(1,1,1,1))
        self.assertEqual(result['f1'],.5); self.assertEqual(result['detection_delay_seconds'],[1])
        self.assertEqual(result['false_alarm_onsets'],1)
        self.assertIsNone(metrics([(0,False)],[])['precision'])
    def test_same_content_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as td:
            p=Path(td); (p/'a').write_bytes(b'clip'); (p/'b').write_bytes(b'clip')
            clips=[dict(path='a',split='tune',source_id='one',fight_intervals=[]),dict(path='b',split='evaluation',source_id='two',fight_intervals=[])]
            with self.assertRaisesRegex(ValueError,'leakage'): validate_manifest(clips,p)
    def test_same_source_excerpts_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as td:
            p=Path(td); (p/'a').write_bytes(b'first'); (p/'b').write_bytes(b'second')
            clips=[dict(path='a',split='tune',source_id='one',fight_intervals=[]),dict(path='b',split='evaluation',source_id='one',fight_intervals=[])]
            with self.assertRaisesRegex(ValueError,'leakage'): validate_manifest(clips,p)
