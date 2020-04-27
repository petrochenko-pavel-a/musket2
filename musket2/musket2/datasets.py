from musket2 import binding_platform,utils
import os
import numpy as np
import tqdm
from sklearn import model_selection as ms
from  torch.utils.data import DataLoader
import torch

class PredictionItem:

    def __init__(self, path, x, y, prediction=None):
        self.x = x
        self.y = y
        self.id = path
        self.prediction = prediction

    def original(self):
        return self

    def rootItem(self):
        return self

    def item_id(self):
        return self.id


class DataSet:

    def __init__(self):
        self.parent = None

    def __getitem__(self, item) -> PredictionItem:
        raise ValueError("Not implemented")

    def get_train_item(self, item) -> PredictionItem:
        return self[item]

    def __len__(self):
        raise ValueError("Not implemented")

    def stratification_group(self,item):
        return 0

    def stratify_classes(self):
        preds=[]
        for i in tqdm.tqdm(range(len(self)), "reading stratification classes " + str(self)):
            preds.append(self.stratification_group(i))
        return np.array(preds)

    def root(self):
        if self.parent is not None:
            return self.parent.root()
        return self


class CompositeDataSet(DataSet):

    def __init__(self, components):
        super().__init__()
        self.components = components
        sum = 0;
        shifts = []
        for i in components:
            sum = sum + len(i)
            shifts.append(sum)
        self.shifts = shifts
        self.len = sum

    def get_train_item(self, item) -> PredictionItem:
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                if hasattr(d, "get_train_item"):
                    return d.get_train_item(i)
                res = d[i]
                res.originalDataSet = d
                return res
            else:
                i = item - self.shifts[j]

    def get_target(self, item):
        i = item
        for j in range(len(self.shifts)):
            provider = self.components[j]
            if item < self.shifts[j]:
                if hasattr(provider, "get_target"):
                    return provider.get_target(i)
                else:
                    return provider[i]
            else:
                i = item - self.shifts[j]

    def __getitem__(self, item):
        for j in range(len(self.shifts)):
            d = self.components[j]
            if item < self.shifts[j]:
                res = d[i]
                res.originalDataSet = d
                return res
            else:
                i = item - self.shifts[j]

    def __len__(self):
        return self.len


class SubDataSet(DataSet):
    def __init__(self, orig, indexes):
        super().__init__()
        self.ds = orig
        self.parent = orig
        self.indexes = indexes

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]

    def get_train_item(self, item: int):
        if hasattr(self.ds, "get_train_item"):
            return self.ds.get_train_item(self.indexes[item])
        else:
            return self.ds[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)

    def get_target(self, item):
        return self.parent.get_target(self.indexes[item])


class DatasetProviderBase(binding_platform.ExtensionBase):
    pass


class WriteableDataSet(DataSet):

    def append(self, item):
        raise ValueError("Not implemented")

    def load(self):
        pass

    def commit(self):
        raise ValueError("Not implemented")

    def blend(self, ds, w=0.5) -> DataSet:
        return self._inner_blend(ds, w)

    def _inner_blend(self, ds, w=0.5):
        raise NotImplementedError("Not supported")


class BufferedWriteableDS(WriteableDataSet):

    def __init__(self, orig, name, dsPath, predictions=None, pickle=False):
        super().__init__()
        if predictions is None:
            predictions = []
        self.parent = orig
        self.name = name
        self.pickle = pickle
        self.predictions = predictions
        self.dsPath = dsPath

    def append(self, item):
        self.predictions.append(item)

    def blend(self, ds, w=0.5) -> DataSet:
        return super().blend(ds, w)

    def load(self):
        if self.dsPath is not None:
            if self.pickle:
                self.predictions=utils.load(self.dsPath)
                self.predictions = np.array(self.predictions)
            else:
                self.predictions=np.load(self.dsPath+".npy")

    def commit(self):
        if self.dsPath is not None:
            if self.pickle:
                utils.save(self.dsPath, self.predictions)
            else:
                np.save(self.dsPath, self.predictions)
        self.predictions = np.array(self.predictions)

    def __len__(self):
        return len(self.parent)

    def _inner_blend(self, ds, w=0.5):
        if isinstance(self.predictions, list):
            q = []
            for i in tqdm.tqdm(range(len(self.predictions))):
                nm = self.predictions[i];
                if isinstance(nm, list):
                    nm1 = ds.predictions[i];
                    ress = []
                    for j in range(len(nm)):
                        ress.append(nm[j] * w + nm1[j] * (1 - w))
                    q.append(ress)
                else:
                    q.append(self.predictions[i] * w + ds.predictions * (1 - w))
            return BufferedWriteableDS(self.parent, self.name, None, q)
        pr = self.predictions * w + ds.predictions * (1 - w)
        return BufferedWriteableDS(self.parent, self.name, None, pr)

    def __getitem__(self, item):
        it = self.parent[item]
        if self.predictions is not None:
            if isinstance(item, slice):
                indSlice = list(range(len(self)))[item]
                for i in range(len(it)):
                    it[i].prediction = self.predictions[indSlice[i]]
            else:
                it.prediction = self.predictions[item]
        return it


class DirectWriteableDS(WriteableDataSet):

    def __init__(self, orig, name, dsPath, count=0):
        super().__init__()
        self.parent = orig
        self.name = name
        self.dsPath = dsPath
        self.count = count

    def append(self, item):
        ip = self.item_path(self.count)
        self.save_item(ip, item)
        self.count += 1

    def commit(self):
        pass

    def __len__(self):
        return len(self.parent)

    def _inner_blend(self, ds, w=0.5):
        return WeightedBlend([self, ds], [w, 1 - w])

    def __getitem__(self, item):
        it = self.parent[item]
        ip = self.item_path(item)
        if os.path.exists(ip):
          it.prediction = self.load_item(ip)
        return it

    def item_path(self, item: int) -> str:
        return f"{self.dsPath}/{item}.npy"

    def save_item(self, path: str, item):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.mkdir(dir)
        np.save(path, item)

    def load_item(self, path: str):
        return np.load(path)


class CompressibleWriteableDS(DirectWriteableDS):

    def __init__(self, orig, name, dsPath, count=0, asUints=True, scale=255):
        super().__init__()
        self.parent = orig
        self.name = name
        self.dsPath = dsPath
        self.count = count
        self.asUints = asUints
        self.scale = scale

    def item_path(self, item: int) -> str:
        return f"{self.dsPath}/{item}.npy.npz"

    def save_item(self, path: str, item):
        dire = os.path.dirname(path)
        if item is None:
            raise ValueError("Should never happen")
        if self.asUints:
            if self.scale <= 255:
                item = (item * self.scale).astype(np.uint8)
            else:
                item = (item * self.scale).astype(np.uint16)
        if not os.path.exists(dire):
            os.mkdir(dire)
        np.savez_compressed(path, item)

    def load_item(self, path: str):
        if self.asUints:
            x = np.load(path)["arr_0.npy"].astype(np.float32) / self.scale
        else:
            x = np.load(path)["arr_0.npy"]
        return x


class WeightedBlend(DataSet):

    def __init__(self, path, nm, weights=None):
        super().__init__()
        self.path = path
        self.nm = nm
        if weights is None:
            weights = []
        if isinstance(path, list):
            v = path
            self.predictions = []
            for path in v:

                self.predictions.append(path)
                if len(weights) < len(self.predictions):
                    weights.append(1)
        sw = sum(weights)
        self.weights = [x / sw for x in weights]
        self.cache = {}

    def __len__(self):
        return len(self.predictions[0])

    def blend(self, ds, w=0.5) -> DataSet:
        return self._inner_blend(ds, w)

    def _inner_blend(self, ds, w=0.5):
        return WeightedBlend([self, ds], [w, 1 - w])

    def __getitem__(self, item) -> PredictionItem:
        z = self.predictions[0][item]
        if z.prediction is None:
            return z
        prs = []
        for i in range(len(self.predictions)):
            v = self.predictions[i]
            w = self.weights[i]
            if v[item].prediction is not None:
                prs.append(v[item].prediction * w)
        pr = np.sum(prs, axis=0)
        r = PredictionItem(z.id, z.x, z.y, pr)
        return r


class DataSplit:

    def __init__(self,train:DataSet,validation:DataSet=None,test:DataSet=None):
        self.train=train
        self.validation=validation
        self.test=test


class KFoldedSplitter:

    def __init__(self, ds:DataSet, folds_count:int, stratified=False, rs=123, validation_split=0.2, test_split=None,
                 test:DataSet=None
                 ):
        indexes = range(len(ds))
        self.rs=rs
        self.stratified=stratified
        self.testDataSet=test
        self.ds=ds
        if stratified:
            self.classes=ds.stratify_classes()
        if test_split is not None and test_split>0:
            ts=self._split(indexes,test_split)
            indexes=ts[0]
            self.test_indexes=ts[1]
        else:
            self.test_indexes=None
        if folds_count == 1:
            if validation_split>0:
                r = self._split(indexes, validation_split)
                self.folds = [r]
            else:
                self.folds=[indexes,[]]
        else:
            if stratified:
                classes = self.classes
                self.kf = ms.StratifiedKFold(folds_count, shuffle=True, random_state=rs)
                self.folds = [v for v in self.kf.split(indexes, classes)]
            else:
                self.kf = ms.KFold(folds_count, shuffle=True, random_state=rs)
                self.folds = [v for v in self.kf.split(indexes)]
        pass

    def __getitem__(self, item):
        return self.fold(item)

    def __len__(self):
        return len(self.folds)

    def _split(self, indexes,   validation_split):
        stratified=self.stratified
        if stratified:
            classes = self.classes
            r = ms.train_test_split(list(indexes), classes, shuffle=True, stratify=classes, random_state=self.rs,
                                    test_size=validation_split)
        else:
            r = ms.train_test_split(list(indexes), shuffle=True, random_state=self.rs, test_size=validation_split)
        return r

    def fold(self,num:int)->DataSplit:
        ds=self.ds
        train=SubDataSet(ds,self.folds[num][0])
        if len(self.folds[num][1])>0:
            val = SubDataSet(ds, self.folds[num][1])
        else:
            val = None
        test=None
        if self.testDataSet is not None:
            test=self.testDataSet
        if self.test_indexes is not None:
            test=SubDataSet(ds,self.test_indexes)
        return DataSplit(train,val,test)


dataset_provider = type("DataSetProvider", (DatasetProviderBase,), {})

class DataOwner:

    def __init__(self,wr,batch_size,num_data_workers=0,shuffle=True,dropLast=True):
        self.wr=wr
        self.batch_size=batch_size
        self.num_data_workers=num_data_workers
        self.shuffle=shuffle
        self.dropLast=dropLast
        pass

    def new_loader(self):
        return DataLoader(self.wr,batch_size=self.batch_size,num_workers=self.num_data_workers,shuffle=self.shuffle,drop_last=self.dropLast)

class BasicWrapper:

    def __init__(self, ds) -> None:
        self.ds = ds

    def __getitem__(self, ind: int):
         row = self.ds[ind]
         return {"x": torch.from_numpy(row.x), "y": torch.from_numpy(row.y)}

    def __len__(self) -> int:
        return len(self.ds)

def create(cfg:any):
    return binding_platform.create_extension(dataset_provider,cfg)


def create_loader(ds,batch_size,num_data_workers=0,shuffle=True,drop_last=True):
    sw=BasicWrapper(ds)
    return DataOwner(sw,batch_size,num_data_workers,shuffle,drop_last).new_loader()

