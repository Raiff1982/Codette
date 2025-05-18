import os
import json
import unittest
from analyze_cocoons1 import analyze_cocoons1
from analyze_cocoons2 import analyze_cocoons2
from analyze_cocoons3 import analyze_cocoons3

class TestAnalyzeCocoons(unittest.TestCase):

    def setUp(self):
        self.folder = './test_cocoons/'
        os.makedirs(self.folder, exist_ok=True)
        self.create_test_cocoons()

    def tearDown(self):
        for fname in os.listdir(self.folder):
            os.remove(os.path.join(self.folder, fname))
        os.rmdir(self.folder)

    def create_test_cocoons(self):
        cocoons = [
            {
                "filename": "test1.cocoon",
                "data": {
                    "quantum_state": [0.1, 0.2],
                    "chaos_state": [0.3, 0.4, 0.5],
                    "run_by_proc": 1,
                    "perspectives": ["perspective1", "perspective2"]
                }
            },
            {
                "filename": "test2.cocoon",
                "data": {
                    "quantum_state": [0.6, 0.7],
                    "chaos_state": [0.8, 0.9, 1.0],
                    "run_by_proc": 2,
                    "perspectives": ["perspective3", "perspective4"]
                }
            }
        ]
        for cocoon in cocoons:
            with open(os.path.join(self.folder, cocoon["filename"]), 'w') as f:
                json.dump({"data": cocoon["data"]}, f)

    def test_analyze_cocoons1(self):
        result = analyze_cocoons1(self.folder)
        self.assertIn("test1.cocoon", result)
        self.assertIn("test2.cocoon", result)

    def test_analyze_cocoons2(self):
        result = analyze_cocoons2(self.folder)
        self.assertIn("test1.cocoon", result)
        self.assertIn("test2.cocoon", result)

    def test_analyze_cocoons3(self):
        result = analyze_cocoons3(self.folder)
        self.assertIn("test1.cocoon", result)
        self.assertIn("test2.cocoon", result)

if __name__ == '__main__':
    unittest.main()
