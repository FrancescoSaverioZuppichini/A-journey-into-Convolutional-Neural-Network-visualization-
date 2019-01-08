import h5py
import torch

from torch.utils.data import Dataset

class PerceptionDataset(Dataset):
    def __init__(self, group, h5_path, transform=None, test=False):
        h5f = h5py.File(h5_path, 'r')

        self.Xs = {i: h5f['bag' + str(i) + '_x'] for i in group}
        self.Ys = {i: h5f['bag' + str(i) + '_y'] for i in group}
        self.lengths = {i: self.Xs[i].shape[0] for i in group}
        self.counts = {i: 0 for i in group}
        self.transform = transform
        self.test = test

        self.Y = []
        self.X = []

        self.length = 0

        for i in group:
            self.X += list(self.Xs[i][:])
            self.Y += list(self.Ys[i][:])

        for el in self.Xs.values():
            self.length += len(el)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item]

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        x = x.permute(2, 0, 1)

        if self.transform is not None:
            x = self.transform(x)


        y[y > 0] = 1.0

        return x, y

    def __len__(self):
        return self.length