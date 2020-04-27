from musket2 import  generic,datasets
from .datasets import create_segmentation_loader
import segmentation_models_pytorch as stm


class PipelineConfig(generic.GenericConfig):

    def  __init__(self,**atrs):
        self.shape = None
        self.batch_size = None
        self.data_workers = 0
        self.shuffle_data = True
        self.drop_last = False
        self.aug = None
        self.cfg = atrs
        self.architecture=None
        self.backbone=None
        self.activation=None
        super().__init__(**atrs)
        def batchPrepare(batch, device, non_blocking):
            return batch["image"].cuda(), batch["label"].cuda()
        self.batchPrepare=batchPrepare

        def outputTransform(y, y1, y2):
            return y1, (y2 > 0.5).type_as(y1)
        self.outputTransform=outputTransform


    def create_writable_ds(self, ds,name, path):
        return datasets.CompressibleWriteableDS(ds,name,path)

    def create_model(self):
        clazz = getattr(stm, self.architecture)
        return clazz(self.backbone,activation=self.activation)

    def create_loader(self,ds):
        return  create_segmentation_loader(ds,self.shape,self.batch_size,self.drop_last,self.aug)

