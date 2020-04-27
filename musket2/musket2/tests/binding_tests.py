import unittest
from musket2 import binding_platform
import torch.optim


class BasicBindingTests(unittest.TestCase):

    def test_adam(self):
        opt=binding_platform.create_extension(torch.optim.Optimizer,"Adam",[torch.ones(10)])
        self.assertIsNotNone(opt)
        opt=binding_platform.create_extension(torch.optim.Optimizer,"adam",[torch.ones(10)])
        self.assertIsNotNone(opt)
        opt = binding_platform.create_extension(torch.optim.Optimizer, "adam", [torch.ones(10)],{ "lr":0.01})
        self.assertIsNotNone(opt)
        d={"lr": 0.2}
        opt = binding_platform.create_extension(torch.optim.Optimizer, "adam", [torch.ones(10)], {"$parent":d})
        self.assertEqual(opt.param_groups[0]["lr"],0.2)

