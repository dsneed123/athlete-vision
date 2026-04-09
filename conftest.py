import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# mediapipe >= 0.10 removed mp.solutions; inject a minimal stub so that
# unittest.mock.patch("mediapipe.solutions.pose.Pose") works in tests.
if "mediapipe.solutions" not in sys.modules:
    import mediapipe as _mp

    _solutions = types.ModuleType("mediapipe.solutions")
    _pose_mod = types.ModuleType("mediapipe.solutions.pose")

    class _Pose:  # minimal stand-in; always replaced by patch() in tests
        def __init__(self, **kwargs):
            pass

        def process(self, image):
            return None

        def close(self):
            pass

    _pose_mod.Pose = _Pose
    _solutions.pose = _pose_mod

    sys.modules["mediapipe.solutions"] = _solutions
    sys.modules["mediapipe.solutions.pose"] = _pose_mod
    _mp.solutions = _solutions
