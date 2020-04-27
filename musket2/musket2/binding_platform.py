from musket2 import introspector
import inspect
import torch

def lookup(dct:dict,name:str):
    if  name in dct:
        return  dct[name]
    if "$parent" in dct:
        return lookup(dct["$parent"],name)
    return None

class ExtensionBase:

    _registry = {}

    def __init__(self,name=None):
        self.name=name
        self.object=None
        pass

    def __call__(self, obj):
        if self.name is None:
            name=obj.__name__
        else:
            name=self.name

        if self.__class__ not  in ExtensionBase._registry:
            ExtensionBase._registry[self.__class__]={}
        ExtensionBase._registry[self.__class__][name]=self
        self.object=obj
        self.meta=introspector.get_meta(obj)
        self.parameters=self.meta["parameters"]

        return obj

    def create(self,kwArgs,defaultObject):
        args=kwArgs.copy()
        if "$parent" in args:
            del args["$parent"]
        argsList=[]
        seqArgs=True
        for p in self.parameters:
            val=lookup(kwArgs,p["name"])
            if val is None and not "defaultValue" in p:
                argsList.append(defaultObject)
            if val is not None:
                if seqArgs:
                    argsList.append(val)
                    del args[p["name"]]
                else:
                    args[p["name"]]=val
            else:
                seqArgs=False
        if len(args)>0:
            return self.object(*argsList,**args)
        else:

            return self.object(*argsList)

def extensions(clazz:type):
    if clazz not in ExtensionBase._registry:
        return []
    return ExtensionBase._registry[clazz]


def extension(clazz:type,name:str)->ExtensionBase:
    if clazz not in ExtensionBase._registry:
        return None
    if name in ExtensionBase._registry[clazz]:
        return ExtensionBase._registry[clazz][name]
    name=name[0].upper()+name[1:]
    if name in ExtensionBase._registry[clazz]:
        return ExtensionBase._registry[clazz][name]
    return None

def register(clazz:type,name,object):
    if clazz not in ExtensionBase._registry:
        ExtensionBase._registry[clazz] = {}
    eb=ExtensionBase(name)
    eb(object)
    ExtensionBase._registry[clazz][name] =eb


def create_extension(clazz:type,name:str,defaultObject=None,cfg={}):

    if isinstance(name,dict):
        mname=name.__iter__().__next__()
        cfg=name[mname]
        name=mname
    ext=extension(clazz,name)
    if ext is not None:
        return ext.create(cfg,defaultObject)
    return None


def register_classes(module, clazz):
    lst = dir(module)
    result = []
    for item in lst:
        if item[0] == '_':
            continue

        attribute = getattr(module, item)

        if inspect.isclass(attribute):
            try:
                if isinstance(clazz, type) and issubclass(attribute, clazz):
                    register(clazz,item,attribute)
            except:
                print("Can not inspect", attribute)
                import traceback
                traceback.print_exc()

register_classes(torch.optim,torch.optim.Optimizer)
register(torch.optim.lr_scheduler._LRScheduler,torch.optim.lr_scheduler.ReduceLROnPlateau.__name__,torch.optim.lr_scheduler.ReduceLROnPlateau)
register_classes(torch.optim.lr_scheduler,torch.optim.lr_scheduler._LRScheduler)