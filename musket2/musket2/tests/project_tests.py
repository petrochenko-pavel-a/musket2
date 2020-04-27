import unittest
import os
import warnings
from musket2 import context,generic,datasets
import imgaug
import numpy as np
import  traceback
context.set_current_project_path(os.path.join(os.path.dirname(__file__),"test_data"))

@datasets.dataset_provider(name="dummy")
class DummDataSet(datasets.DataSet):
    def __len__(self):
        return 100

    def __getitem__(self, item):
        return datasets.PredictionItem(item, np.zeros((200,200,3)), np.zeros((200,200),dtype=np.bool))



class BasicDataTests(unittest.TestCase):

    def test_split0(self):
        cfg=generic.parse("simple_seg")
        cfg.fit()
        print(cfg)

    def test_mnist(self):
        pass
        #cfg=generic.parse("mnist")
        #cfg.fit()