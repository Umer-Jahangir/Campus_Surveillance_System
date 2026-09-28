"""Regressions for decoder process memory and temporal IPC ownership."""
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import worker


class FocusedRuntimeTests(unittest.TestCase):
    def test_decoder_import_does_not_load_inference_runtimes(self):
        code = """
import sys
sys.path.insert(0, 'src')
import worker
from fight_detector_onnx import DisabledPoseClassifier
assert not ({'torch', 'ultralytics', 'supervision', 'onnxruntime'} & sys.modules.keys())
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                       capture_output=True, timeout=30)

    def test_live_temporal_clip_transferred_once_per_window(self):
        stop = threading.Event()
        ready = threading.Event()
        ready.set()
        frames, clips = queue.Queue(200), queue.Queue(20)

        class Capture:
            count = 0
            def set(self, *args): pass
            def isOpened(self): return True
            def get(self, prop): return 15
            def release(self): pass
            def read(self):
                self.count += 1
                if self.count == 120:
                    stop.set()
                return True, np.full((24, 32, 3), self.count, np.uint8)

        cap = Capture()
        with patch('worker.cv2.VideoCapture', return_value=cap), \
             patch('worker.psutil.Process'), \
             patch('worker.assign_cores_hybrid', return_value=[0]), \
             patch('worker.time.monotonic', side_effect=lambda: cap.count / 15):
            worker.decoder_worker(0, 'rtsp://fixture', (24, 32, 3), frames,
                                  stop, ready, 1, 1, clips)
        packets = list(frames.queue)
        windows = list(clips.queue)
        self.assertEqual(len(packets), 120)
        self.assertTrue(all(p['scene_clip'] is None for p in packets))
        self.assertIsNone(windows[0]['clip'])  # Explicit initial warmup/reset.
        complete = [p['clip'] for p in windows if p['clip'] is not None]
        self.assertGreaterEqual(len(complete), 3)
        self.assertLessEqual(len(complete), 4)
        self.assertEqual(len({c['end'] for c in complete}), len(complete))
        self.assertEqual(complete[0]['frames'].shape, (16, 224, 224, 3))
        first = complete[0]['frames'].copy()
        complete[-1]['frames'][:] = 0
        np.testing.assert_array_equal(first, complete[0]['frames'])
        self.assertLessEqual(complete[-1]['end'], packets[-1]['source_time'])


if __name__ == '__main__':
    unittest.main()
