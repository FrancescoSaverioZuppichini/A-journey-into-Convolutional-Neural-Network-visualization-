import cv2
import numpy as np
import torch

from torch.nn import ReLU
from torch.autograd import Variable
from .Base import Base
from torch.nn import AvgPool2d, Conv2d, Linear, ReLU, MaxPool2d, BatchNorm2d
import torch.nn.functional as F

from .utils import tensor2cam, module2traced, imshow

class GradCam(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handles = []
        self.gradients = None
        self.conv_outputs = None

    def store_outputs_and_grad(self, layer):
        def store_grads(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def store_outputs(module, input, outputs):
            if module == layer:
                self.conv_outputs = outputs

        self.handles.append(layer.register_forward_hook(store_outputs))
        self.handles.append(layer.register_backward_hook(store_grads))

    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_out[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))


    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, input_image, layer, guide=False, target_class=None, postprocessing=lambda x: x, regression=False):
        self.clean()
        self.module.zero_grad()

        if layer is None:
            modules = module2traced(self.module, input_image)
            for i, module in enumerate(modules):
                if isinstance(module, Conv2d):
                    layer = module

        self.store_outputs_and_grad(layer)

        if guide: self.guide(self.module)

        input_var = Variable(input_image, requires_grad=True).to(self.device)
        predictions = self.module(input_var)

        if target_class is None: values, target_class = torch.max(predictions, dim=1)
        if regression: predictions.backward(gradient=target_class, retain_graph=True)
        else:
            target = torch.zeros(predictions.size()).to(self.device)
            target[0][target_class] = 1
            predictions.backward(gradient=target, retain_graph=True)

        with torch.no_grad():
            avg_channel_grad = F.adaptive_avg_pool2d(self.gradients.data, 1)
            self.cam = F.relu(torch.sum(self.conv_outputs[0] * avg_channel_grad[0], dim=0))

            image_with_heatmap = tensor2cam(postprocessing(input_image.squeeze().cpu()), self.cam)

        self.clean()

        return image_with_heatmap.unsqueeze(0), { 'prediction': target_class}


