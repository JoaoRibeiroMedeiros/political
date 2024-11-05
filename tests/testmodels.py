import unittest
import pandas as pd
from crop import crop_statements_until_t
from src.models import SimulateStatement, Model, ModelStats

class TestModels(unittest.TestCase):

    def setUp(self):
        self.tau = crop_statements_until_t


