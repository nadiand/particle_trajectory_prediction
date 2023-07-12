import torch
import numpy as np
import math
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

    def __init__(self, data, train=True, labels=None, shuffle=True, sort_data=False, sort_targets=False):
        if shuffle and sort_data:
            raise Exception("Only one out of sort and shuffle can be True at a time!")
        
        self.data = data.fillna(value=PAD_TOKEN)
        self.train = train
        self.targets = labels
        if self.train:
            self.targets = self.targets.fillna(value=PAD_TOKEN)
        self.shuffle = shuffle # True now because synthesized data has order, irl probably False
        self.sort_data = sort_data
        self.sort_targets = sort_targets
        self.total_events = self.__len__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Load a single event (row)
        event = self.data.iloc[[idx]].values.tolist()[0]
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
                targets = np.sort(targets)
            targets = torch.tensor(targets).float()
        
            # Obtain the track ("class") each hit belongs to
            track_classes = []
            for i, coord in enumerate(x):
                if coord != PAD_TOKEN:
                    # If an actual detected hit, 1-hot encode its "class"
                    track_lbl = [0]*MAX_NR_TRACKS
                    track_lbl[int(i/NR_DETECTORS)] = 1
                else:
                    # If a padded hit, its "class" vector is made of PAD_TOKEN
                    track_lbl = [PAD_TOKEN]*MAX_NR_TRACKS
                track_classes.append(track_lbl)

        # Shuffle data (and corresponding classes) if specified
        if self.shuffle:
            shuffled_indices = torch.randperm(len(data))
            data = data[shuffled_indices]
            track_classes = torch.tensor(track_classes).float()
            track_classes = track_classes[shuffled_indices]

        # Sort data (and corresponding classes) if specified
        if self.sort_data:
            if DIM == 2:
                # Sort the data by angle
                angles = [math.asin(coord[0]/math.sqrt(coord[0]**2 + coord[1]**2)) for coord in data]
                _, indices = torch.sort(torch.tensor(angles).float())
            if DIM == 3:
                # Sort the data by distance from the origin
                pythagorean = [coord[0]**2 + coord[1]**2 + coord[2]**2 for coord in data]
                _, indices = torch.sort(torch.tensor(pythagorean).float())
            data = data[indices] 
            track_classes = torch.tensor(track_classes).float()
            track_classes = track_classes[indices]
        
        del event
        return event_id, data, targets, track_classes
    