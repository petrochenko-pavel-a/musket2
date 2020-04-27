import imageio
import os
from musket2.datasets import DataSet,PredictionItem,DataOwner
from musket2 import utils
import numpy as np
import imgaug as iaa
import torch
import  traceback


class SimpleDirBasedDataSet(DataSet):

    def __init__(self, path, in_ext="png", out_ext="png"):
        path = utils.normalize_path(path)
        self.path = path
        ldir = os.listdir(path)
        if ".DS_Store" in ldir:
            ldir.remove(".DS_Store")
        self.ids = [x[0:x.index('.')] for x in ldir if "mask_floor" in x]
        self.in_ext = in_ext
        self.out_ext = out_ext
        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        in_ext = self.in_ext
        image = imageio.imread(os.path.join(self.path, self.ids[item] + "." + in_ext))
        mask = imageio.imread(os.path.join(self.path, self.ids[item] + "." + in_ext + "_mask_floor.png"))
        image = image.astype(np.uint8)
        p = PredictionItem(self.ids[item] + str(), image[:, :, 0:3], mask[:, :, 0].astype(np.bool))
        return p



class SegmentationWrapper:

    def __init__(self, ds, shape,aug=None,nc=1) -> None:
        self.ds = ds
        self.nc=nc
        rs=iaa.augmenters.Resize({"height": shape[0], "width": shape[1]})
        self.shape=shape
        if aug is not None:
            self.aug:iaa.augmenters.Augmenter=iaa.augmenters.Sequential([aug,rs])
        else:
            self.aug =rs

    def __getitem__(self, ind: int):
        try:
            row = self.ds[ind]
            image_aug, segmap_aug= self.aug(image=row.x,segmentation_maps=iaa.SegmentationMapsOnImage(row.y,shape=(row.y.shape[0],row.y.shape[1],1)))

            image = torch.from_numpy(image_aug.astype(np.float32)).permute((2, 0, 1))
            mask = torch.from_numpy(segmap_aug.arr.astype(np.float32)).permute((2, 0, 1))
            return {"image": image, "label": mask}
        except:
            traceback.print_exc()
            image = torch.zeros(( self.shape[2], self.shape[0], self.shape[1]))
            mask =  torch.zeros(( self.nc, self.shape[0], self.shape[1]))
            return {"image": image, "label": mask}

    def __len__(self) -> int:
        return len(self.ds)




def create_segmentation_loader(ds,shape,batch_size,num_data_workers=0,shuffle=True,drop_last=True,augmentation=None,nc=1):
    sw=SegmentationWrapper(ds,shape,aug=augmentation,nc=nc)
    return DataOwner(sw,batch_size,num_data_workers,shuffle,drop_last).new_loader()
