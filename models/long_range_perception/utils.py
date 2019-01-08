import torch

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import PerceptionDataset
import matplotlib.pyplot as plt

BATCH_SIZE = 128
N_WORKERS = 14
H5_PATH = '/Users/vaevictis/Documents/Project/Advance-Topics-In-Machine-Learning-Project/data_many_dist_fixed_step.h5'
GROUP = np.arange(1)
EPOCHES = 20

def get_dl(*args, **kwargs):
    ds = PerceptionDataset(*args, **kwargs)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=N_WORKERS, drop_last=True, shuffle=not kwargs['test'])

    return dl, ds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def un_normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

post_processing = Compose([un_normalize])


def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()
