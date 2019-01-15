import torch

from torch.nn import AvgPool2d, Conv2d, Linear, ReLU
from torch.nn.functional import softmax

from .Base import Base

from .utils import module2traced, imshow, tensor2cam

import torch.nn.functional as F


class ClassActivationMapping(Base):
    """
    Based on Learning Deep Features for Discriminative Localization (https://arxiv.org/abs/1512.04150).
    Be aware,it requires feature maps to directly precede softmax layers.
    It will work for resnet but not for alexnet for example
    """

    def __call__(self, inputs, layer, target_class=None, postprocessing=lambda x: x, guide=False):
        modules = module2traced(self.module, inputs)
        last_conv = None
        last_linear = None

        for i, module in enumerate(modules):
            if isinstance(module, Conv2d):
                last_conv = module
            if isinstance(module, AvgPool2d):
                pass
            if isinstance(module, Linear):
                last_linear = module

        def store_conv_outputs(module, inputs, outputs):
            self.conv_outputs = outputs

        last_conv.register_forward_hook(store_conv_outputs)

        predictions = self.module(inputs)

        if target_class == None: _, target_class = torch.max(predictions, dim=1)
        _, c, h, w = self.conv_outputs.shape
        # get the weights relative to the target class
        fc_weights_class = last_linear.weight.data[target_class]
        # sum up the multiplication of each weight w_k for the relative channel in the last
        # convolution output
        cam = fc_weights_class @ self.conv_outputs.view((c, h * w))
        cam = cam.view(h, w)

        with torch.no_grad():
            image_with_heatmap = tensor2cam(postprocessing(inputs.cpu().squeeze()), cam)

        return image_with_heatmap.unsqueeze(0), { 'prediction': target_class }