import torch

from core import *
from core.utils import image_net_postprocessing
from PIL import Image

from torchvision.models import alexnet, vgg16, resnet18, resnet152
from torchvision.transforms import ToTensor, Resize, Compose

import matplotlib.pyplot as plt
from core.utils import image_net_postprocessing, image_net_preprocessing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create a model
model = resnet152(pretrained=True)
# print(model)
cat = Image.open("/home/francesco/Desktop/cat.jpg")
# resize the image and make it a tensor
input = Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(cat)
# add 1 dim for batch
input = input.unsqueeze(0)
# call mirror with the input and the model
layers = list(model.children())
# layer = layers[4][2]
# print(layer)

def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()

model.eval()

vis = GradCam(model.to(device), device)
img = vis(input.to(device), None,
          target_class=None,
          postprocessing=image_net_postprocessing,
          guide=False)


with torch.no_grad():
    imshow(img[0])