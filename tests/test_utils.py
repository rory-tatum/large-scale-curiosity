import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curious.utils import *

class TestUtils(unittest.TestCase):
    def test_guess_available_gpus(self):
        self.assertListEqual([0, 1, 2, 3], guess_available_gpus(4))
        self.assertListEqual([0], guess_available_gpus())
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()