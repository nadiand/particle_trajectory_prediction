import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from global_constants import *

def cart2cyl(x, y, z=None):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi, z) if z is not None else (rho, phi)


def sort_by_angle(cartesian_coords):
    dist_coords = np.array(cartesian_coords)
    distances = np.round(np.linalg.norm(dist_coords, axis=1))
    # Sort first by rho, round the rho, then sort by phi (sorting by the angle on decoder)
    cylindrical_coords = [cart2cyl(*coord) for coord in cartesian_coords]
    sorted_indices = np.lexsort((list(zip(*cylindrical_coords))[1], distances))
    sorted_cartesian_coords = [cartesian_coords[i] for i in sorted_indices]
    return sorted_cartesian_coords


# def earth_mover_distance(y_true, y_pred):
#     distance = torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
#     return torch.mean(torch.mean(distance, dim=tuple(range(1, distance.ndim))))

def pad(tens):
    tens[0] = nn.ConstantPad1d((0, PAD_LEN_DATA - tens[0].shape[0]), PAD_TOKEN)(tens[0])
    all_padded = pad_sequence(tens, batch_first=False, padding_value=PAD_TOKEN)
    return all_padded

def collate_fn(batch):
    event_ids = []
    xs, ys, zs = [], [], []
    labels = []

    for sample in batch:
        event_ids.append(sample[0])
        xs.append(sample[1])
        ys.append(sample[2])
        zs.append(sample[3])
        labels.append(sample[5])

    xs_pad, ys_pad, zs_pad = pad(xs), pad(ys), pad(zs)

    # labels[0] = nn.ConstantPad1d((0, MAX_NR_TRACKS - labels[0].shape[0]), PAD_TOKEN)(labels[0])
    # labels_pad = pad_sequence(labels, batch_first=False, padding_value=PAD_TOKEN)
    # # Convert the lists to tensors, except for the event_id since it might not be a tensor
    # xs[0] = nn.ConstantPad1d((0, PAD_LEN_DATA - xs[0].shape[0]), PAD_TOKEN)(xs[0])
    # ys[0] = nn.ConstantPad1d((0, PAD_LEN_DATA - ys[0].shape[0]), PAD_TOKEN)(ys[0])
    # zs[0] = nn.ConstantPad1d((0, PAD_LEN_DATA - zs[0].shape[0]), PAD_TOKEN)(zs[0])

    # xs_pad = pad_sequence(xs, batch_first=False, padding_value=PAD_TOKEN)
    # ys_pad = pad_sequence(ys, batch_first=False, padding_value=PAD_TOKEN)
    # zs_pad = pad_sequence(zs, batch_first=False, padding_value=PAD_TOKEN)
    x = torch.stack((xs_pad, ys_pad, zs_pad), dim=1)

    # Return the final processed batch
    return event_ids, x.transpose(1,2), labels

class HitsDataset(Dataset):

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
            labels = event_labels[0::DIM]
            labels = np.sort(labels)
            if self.to_tensor:
                labels = torch.tensor(labels).float()

        # the starting indices might be off ! todo
        x = event[0::DIM+1]
        y = event[1::DIM+1]
        z = event[2::DIM+1] if DIM == 3 else [PAD_TOKEN] * len(x)

        # normalise
        if self.normalize:
            self.apply_norm(x)
            self.apply_norm(y)
            self.apply_norm(z)

        convert_list = []
        for i in range(len(x)):
            if DIM == 2:
                convert_list.append((x[i], y[i]))
            else: #dim==3
                convert_list.append((x[i], y[i], z[i]))

        sorted_coords = sort_by_angle(convert_list)

        if DIM == 2:
            x, y = zip(*sorted_coords)
        else:
            x, y, z = zip(*sorted_coords)

        # convert to tensors
        if self.to_tensor:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
            z = torch.tensor(z).float()

        del event
        return event_id, x, y, z, labels
    