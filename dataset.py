import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
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


def custom_collate(batch):
    event_ids = []
    xs = []
    ys = []
    zs = []
    tracks = []
    labels = []

    for b in batch:
        # Assuming z (b[3]) is the variable that can be None
        if b[3] is not None:
            event_ids.append(b[0])
            xs.append(b[1])
            ys.append(b[2])
            zs.append(b[3])
            tracks.append(b[4])
            labels.append(b[5])

    # Convert the lists to tensors, except for the event_id since it might not be a tensor
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    zs = torch.stack(zs)
    tracks = torch.stack(tracks)
    labels = torch.stack(labels)

    # Return the final processed batch
    return event_ids, xs, ys, zs, tracks, labels

class HitsDataset(Dataset):

    def __init__(self, data, train, labels=None, to_tensor=True, normalize=True, shuffle=False):
        self.data = data
        self.labels = labels
        self.train = train
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
    
if __name__ == '__main__':
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    print(dataset.__getitem__(1))