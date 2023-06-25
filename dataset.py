import torch
import numpy as np
from torch.utils.data import Dataset
from global_constants import *


class HitsDataset(Dataset):
    # it assumes that we get csvs as input that we can treat like dataframes
    # also assumes there's no actual index column saved into the csv file and the 
    # indices will be just assigned as 0...N implicitly without it being a column on its own

    def __init__(self, data, train, labels=None, to_tensor=True, normalize=True, shuffle=False):
        self.data = data.fillna(value=PAD_TOKEN)
        self.train = train
        self.labels = labels
        if self.train:
            self.labels = self.labels.fillna(value=PAD_TOKEN)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.total_events = self.__len__()
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def apply_norm(X):
        pass #todo not implemented so dont normalize!!

    def __getitem__(self, idx):
        # load event
        event = self.data.iloc[[idx]].values.tolist()[0]
        # todo: add an actual column with the ID, do not just use the pd id because it's not that generalizable; so that the following is possible:
        # id = event[0]
        event_id = idx

        if self.train:
            event_labels = self.labels.iloc[[idx]].values.tolist()[0]
            if DIM == 2:
                labels = event_labels[0::DIM]
            else: #dim==3
                labels = [] 
                # TODO check if this is correct
                for i in range(0, len(event_labels), DIM):
                    labels.append(event_labels[i])
                    labels.append(event_labels[i+1])
            labels = np.sort(labels)
            # labels = list(filter(lambda value: value != PAD_TOKEN, labels))
            if self.to_tensor:
                labels = torch.tensor(labels).float()
        
        x = event[0::DIM+1]
        y = event[1::DIM+1]
        z = event[2::DIM+1] if DIM == 3 else [PAD_TOKEN] * len(x)

        # normalise
        if self.normalize:
            self.apply_norm(x)
            self.apply_norm(y)
            self.apply_norm(z)

        shuffled_indices = np.arange(0, len(x))
        np.random.shuffle(shuffled_indices)
        shuffled_x, shuffled_y, shuffled_z = [], [], []
        for i in shuffled_indices:
            shuffled_x.append(x[i])
            shuffled_y.append(y[i])
            shuffled_z.append(z[i])

        # convert to tensors
        if self.to_tensor:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
            z = torch.tensor(z).float()

        del event
        return event_id, x, y, z, labels
    