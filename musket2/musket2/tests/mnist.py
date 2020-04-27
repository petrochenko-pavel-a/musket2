import unittest
import os
import warnings
from musket2 import context,generic,datasets
import imgaug
import numpy as np
import  traceback
context.set_current_project_path(os.path.join(os.path.dirname(__file__),"test_data"))
cfg=generic.parse("mnist")
cfg.fit()
#res=cfg.predictions("validation",0,0)
#print(res)