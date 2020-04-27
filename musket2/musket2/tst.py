import torch.optim
import inspect


for v in dir(torch.optim):
    inspect.isclass(v)


torch.optim.Adam([torch.ones((10))])
torch.optim.lr_scheduler.ReduceLROnPlateau