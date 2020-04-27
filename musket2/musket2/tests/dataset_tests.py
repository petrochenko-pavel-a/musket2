import unittest
from musket2 import datasets
import imgaug


class DummDataSet(datasets.DataSet):
    def __len__(self):
        return 100

    def __getitem__(self, item):
        return datasets.PredictionItem(item, 0, 0)



class BasicDataTests(unittest.TestCase):

    def test_split0(self):
        ds=DummDataSet()
        split=datasets.KFoldedSplitter(ds, folds_count=1)
        pnm=split[0]
        self.assertTrue(len(pnm.train)==80)
        self.assertTrue(len(pnm.validation) == 20)
        self.assertTrue(len(split)==1)
        pass

    def test_split1(self):
        ds=DummDataSet()
        split=datasets.KFoldedSplitter(ds, folds_count=1, test_split=0.2)
        pnm=split[0]
        self.assertEqual(len(pnm.train),64)
        self.assertEqual(len(pnm.test), 20)
        self.assertEqual(len(pnm.validation),16)
        self.assertTrue(len(split)==1)
        pass

    def test_split2(self):
        ds=DummDataSet()
        split=datasets.KFoldedSplitter(ds, folds_count=5, test_split=0.2)
        pnm=split[0]
        self.assertEqual(len(pnm.train),64)
        self.assertEqual(len(pnm.test), 20)
        self.assertEqual(len(pnm.validation),16)
        self.assertTrue(len(split)==5)
        pass