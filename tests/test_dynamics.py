import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curious.dynamics import Dynamics, UNet

class TestUtils(unittest.TestCase):
    def test_does_file_compile(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()