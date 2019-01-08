import torch

from .Base import Base
from torch.nn import ReLU
from torch.autograd import Variable
from torchvision.transforms import *
from .utils import convert_to_grayscale

class SaliencyMap(Base):
    """
    Simonyan, Vedaldi, and Zisserman, “Deep Inside Convolutional Networks: Visualising Image Classification Models
    and Saliency Maps”, ICLR Workshop 2014
    https://arxiv.org/abs/1312.6034
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.handles = []
        self.stored_grad = False
        self.rgb2grey = Compose([ToPILImage(), Grayscale(), ToTensor()])

    def store_first_layer_grad(self):

        def hook_grad_input(module, inputs, outputs):
            # stored only for the first time -> first layer
            if not self.stored_grad:
                self.handles.append(module.register_backward_hook(store_grad))
                self.stored_grad = True

        def store_grad(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        for module in self.module.modules():
            self.handles.append(module.register_forward_hook(hook_grad_input))


    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))

    def __call__(self, input_image, layer, guide=False, target_class=None, regression=False):
        self.stored_grad = False
        self.module.zero_grad()

        self.clean()
        if guide: self.guide(self.module)

        input_image = Variable(input_image, requires_grad=True).to(self.device)

        self.store_first_layer_grad()

        predictions = self.module(input_image)

        if target_class is None: values, target_class = torch.max(predictions, dim=1)

        if regression:
            predictions.backward(gradient=target_class, retain_graph=True)
        else:
            target = torch.zeros(predictions.size()).to(self.device)
            target[0][target_class] = 1
            predictions.backward(gradient=target, retain_graph=True)


        image = self.gradients.data.cpu().numpy()[0]
        #
        image = convert_to_grayscale(image)
        image = torch.from_numpy(image).to(self.device)

        self.clean()

        return image.unsqueeze(0), { 'prediction': target_class }

