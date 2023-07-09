import torch
import numpy as np
from torch.utils.data import Dataset

from global_constants import *


class HitsDataset(Dataset):
    '''
    A dataset for the hits with rows corresponding to the number of events, and 
    columns corresponding to the x,y(,z) coordinates of the 5 hits of every
    trajectory in an event and an identifier for each point's belonging to which
    track. The targets contain the trajectory parameter(s) of every track of 
    every event.

    This implementation assumes that it receives as input 2 pandas DataFrames 
    loaded from a csv file with the following structures:
    x_1 y_1 (z_1) trackID_1 x_2 y_2 ... x_n y_n (z_n) trackID_n
    and
    angle1_1 (angle2_1) trackID_1 ... angle1_n (angle2_n) trackID_n
    for the data itself and the targets respectively.
    '''

    def __init__(self, data, train, labels=None, shuffle=True, sort=False):
        self.data = data.fillna(value=PAD_TOKEN)
        self.train = train
        self.targets = labels
        if self.train:
            self.targets = self.targets.fillna(value=PAD_TOKEN)
        self.shuffle = shuffle # True now because synthesized data has order, irl probably should be False
        self.sort_targets = sort
        self.total_events = self.__len__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Load a single event (row)
        event = self.data.iloc[[idx]].values.tolist()[0]
        # todo: add an actual column with the ID, do not just use the pd id because it's not that generalizable; so that the following is possible:
        # id = event[0]
        event_id = idx

        # Obtain the coordinates of each hit
        x = event[0::DIM+1]
        y = event[1::DIM+1]
        z = event[2::DIM+1] if DIM == 3 else None

        # Convert to tensors and create a single matrix for the hits data
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        if z is not None:
            z = torch.tensor(z).float()
            data = torch.stack((x, y, z), dim=1)
        else:
            data = torch.stack((x, y), dim=1)

        nr_events = int(sum([1 for e in x if e != PAD_TOKEN])/NR_DETECTORS)
        real_len = sum([1 for e in x if e != PAD_TOKEN])
        # print(nr_events)

        if self.train:
            # If training, i.e. dataset has targets, obtain the track parameters
            track_params = self.targets.iloc[[idx]].values.tolist()[0]
            if DIM == 2:
                targets = track_params[0::DIM]
            else: #dim==3
                targets = [] 
                for i in range(0, len(track_params), DIM):
                    targets.append((track_params[i], track_params[i+1]))
            if self.sort_targets:
                targets = np.sort(targets) #TODO how does this work for 3d data?
            targets = torch.tensor(targets).float()
        
            # Also obtain the track ("class") each hit belongs to
            track_classes = []
            label = 0
            for coord in x:
                if coord != PAD_TOKEN:
                    # If an actual detected hit, 1-hot encode its "class"
                    track_lbl = [0]*MAX_NR_TRACKS
                    track_lbl[int(label/NR_DETECTORS)] = 1
                else:
                    # If a padded hit, its "class" vector is made of PAD_TOKEN
                    track_lbl = [PAD_TOKEN]*MAX_NR_TRACKS
                track_classes.append(track_lbl)
                label += 1

        # Shuffle data (and corresponding classes) if specified
        if self.shuffle:
            shuffled_indices = np.arange(0, len(data))
            np.random.shuffle(shuffled_indices)
            shuffled_data, shuffled_track_classes = [], []
            for i in shuffled_indices:
                shuffled_data.append(data[i].numpy())
                shuffled_track_classes.append(track_classes[i])
            data = shuffled_data
            track_classes = shuffled_track_classes

        # should targets also be shuffled? TODO if there's sorting of them obv not 
        # but should we sort them :D:D:D
        
        # Convert to tensors
        if self.train:
            track_classes = torch.tensor(track_classes).float()
        data = torch.tensor(np.array(data)).float()
        
        del event
        return event_id, data, targets, track_classes, real_len
    # TODO real_len might be useless and hopefully it is, you should remove it