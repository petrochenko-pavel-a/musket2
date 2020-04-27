import torch
from musket2 import binding_platform

loss_function=type("Loss",(binding_platform.ExtensionBase,),{})

binding_platform.register(loss_function,"binary_crossentropy",torch.nn.BCELoss)
binding_platform.register(loss_function,"categorical_crossentropy",torch.nn.CrossEntropyLoss)
binding_platform.register(loss_function,"l2",torch.nn.MSELoss)
binding_platform.register(loss_function,"l1",torch.nn.L1Loss)


def create(cfg:any):
    return binding_platform.create_extension(loss_function,cfg)