import tqdm
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

from .model import SimpleCNN
from .utils import *

model  = torch.load('./model.pt', map_location=lambda storage, loc: storage)
test_dl, test_ds = get_dl(np.arange(9, 11), H5_PATH, test=True)

# TEST
tot_loss = torch.zeros(1).to(device)
bar = tqdm.tqdm(enumerate(test_dl))
i = 0

model.eval()
rng = np.random.RandomState(13)  # 13 lucky number

distances = list(range(0, 65, 1))
n_dist = len(distances)

auc_array = []

def eval():
    model = SimpleCNN().to(device)

    criterion = nn.MSELoss()

    tot_loss = torch.zeros(1).to(device)
    bar = tqdm.tqdm(enumerate(test_dl))

    for n_batch, (x, y) in bar:

        x, y = x.to(device), y.to(device)

        mask = y != -1

        mask = mask.to(device).float()

        y_ = model(x)
        loss = criterion(y_ * mask, y * mask)


        with torch.no_grad():
            tot_loss += loss

        bar.set_description('loss={:.4f}'.format((tot_loss / (n_batch + 1)).cpu().item()))

    print((tot_loss / (n_batch + 1)).cpu().item())

def make_auc_map():
    for i in range(1):
        with torch.no_grad():
            x, y = test_ds[500]

            x, y = x.unsqueeze(0), y.unsqueeze(0),
            accs = 0

            x, y = x.to(device), y.to(device)

            mask = y != -1

            mask = mask.cpu().numpy()
            # mask[y == -1] = 0
            y_ = model(x)

            y_ = y_.cpu().numpy()
            y = y.cpu().numpy()

            print(mask.shape)
            # for y_1, y1 in zip(y_, y):
                # y1 = np.expand_dims(y1, axis=0)
                # y_1 = np.expand_dims(y_1, axis=0)

            aucs = np.zeros([n_dist, 5])
            y_1 = y
            y1=y_
            for i, d in enumerate(distances):
                for j in range(5):
                    print(mask[:,i * 5 + j])
                    indices = np.where(mask[i * 5 :  i * 5  +j])
                    print(indices, 'indices')

                    yc_1 = y_1[indices, i * 5 + j]
                    yc1 = y1[indices, i * 5 + j]

                    if len(indices[0]) > 0:
                        indices = (rng.choice(indices[0], len(indices[0])),)
                    try:
                        yc1, yc_1 = yc1.tolist()[0],  yc_1.tolist()[0]
                        auc = roc_auc_score(yc1, yc_1)
                        print(auc, 'ddddd')
                    except ValueError as e:
                        auc = 0.5
                    aucs[n_dist - 1 - i, j] = auc
                break
            auc_array.append(aucs)

        # AUC MAP
        mean_auc = np.mean(auc_array, axis=0)
        std_auc = np.std(auc_array, axis=0)

        dist_labels = ['%.0f' % d for d in np.flipud(distances)]
        dist_labels = [d for i, d in enumerate(dist_labels) if i % 2 == 0]
        colors = ['aqua', 'darkorange', 'deeppink', 'cornflowerblue', 'green']

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        sns.heatmap(mean_auc * 100, cmap='gray', annot=True, vmin=50, vmax=100, annot_kws={"color": "white"})
        ax.set_xticklabels(['left', '', 'center', '', 'right'])
        ax.set_yticklabels(dist_labels, rotation=0)
        plt.xlabel('Sensors')
        plt.ylabel('Distance [cm]')
        plt.title('Area Under the pReceiver Operating Characteristic Curve')
        plt.show()

eval()