from musket2 import binding_platform

final_metric=type("final_metric",(binding_platform.ExtensionBase,),{})
metric=type("metric",(binding_platform.ExtensionBase,),{})


def binaryAccuracy(d):
    return 0.4