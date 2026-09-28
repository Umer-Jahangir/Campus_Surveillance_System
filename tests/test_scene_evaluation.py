import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scene_evaluation import event_metrics

class SceneEvaluationTests(unittest.TestCase):
    def test_provisional_labels_cannot_score(self):
        with self.assertRaisesRegex(ValueError,'Confirmed'):event_metrics(10,[],[])
    def test_long_alert_does_not_detect_two_fights(self):
        m=event_metrics(100,[(5,10),(20,25)],[(7,30)],confirmed=True)
        self.assertEqual(m['matched_events'],1);self.assertEqual(m['missed_fights'],1)
    def test_duplicate_alerts_and_hours(self):
        m=event_metrics(3600,[(5,10),(30,40)],[(6,7),(8,9),(100,101)],confirmed=True)
        self.assertEqual(m['false_alerts'],2);self.assertEqual(m['missed_fights'],1)
        self.assertEqual(m['false_alerts_per_camera_hour'],2)
        self.assertEqual(m['time_to_first_alert_seconds'],[1,None])
    def test_warmup_miss_and_negative_no_alert_undefined_precision(self):
        m=event_metrics(10,[(1,2)],[],confirmed=True)
        self.assertEqual(m['event_recall'],0);self.assertIsNone(m['event_precision'])
        m=event_metrics(10,[],[],confirmed=True)
        self.assertIsNone(m['event_recall']);self.assertEqual(m['false_alerts_per_camera_hour'],0)

if __name__=='__main__':unittest.main()
