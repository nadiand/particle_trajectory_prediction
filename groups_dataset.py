import torch
from torch.utils.data import Dataset

from global_constants import *


class GroupsDataset(Dataset):
    '''
    A dataset for the groups of hits forming trajectories, with rows corresponding 
    to x,y(,z) coordinates. The targets contain the "class" label of each point.

    This implementation assumes that it receives as input 2 pandas DataFrames 
    loaded from a csv file with the following structures:
    x_1 y_1 (z_1) trackID_1 x_2 y_2 ... x_n y_n (z_n) trackID_n
    and
    angle1_1 (angle2_1) trackID_1 ... angle1_n (angle2_n) trackID_n
    for the data itself and the targets respectively.
    '''
    def __init__(self, data, train, labels=None):
        self.data = data.fillna(value=PAD_TOKEN)
        self.train = train
        self.targets = labels
        if self.train:
            self.targets = self.targets.fillna(value=PAD_TOKEN)
        self.total_events = self.__len__()
        self.split_events()

    def split_events(self):
        '''
        Splits the complete hits data into groups of hits belonging to the same
        trajectory, and the corresponding track parameter(s). Ignores the padded
        coordinates. 
        '''
        split_data = []
        for i in range(self.total_events):
            # Get an event
            row = self.data.iloc[[i]].values.tolist()[0]
            for j in range(0, len(row), NR_DETECTORS*(DIM+1)):
                if row[j] == PAD_TOKEN:
                    break
                # Grab 5 consequtive coordinates that are from the same track
                x = row[j:j+NR_DETECTORS*(DIM+1):DIM+1]
                y = row[j+1:j+NR_DETECTORS*(DIM+1)+1:DIM+1]
                z = row[j+2:j+NR_DETECTORS*(DIM+1)+2:DIM+1] if DIM == 3 else None
                # Group them and add them to the list
                entry = []
                if z is not None:
                    for i in range(len(x)):
                        entry.append([x[i], y[i], z[i]])
                else:
                    for i in range(len(x)):
                        entry.append([x[i], y[i]])
                split_data.append(entry)
        self.data = split_data

        if self.train:
            # If it's a training dataset, process the targets as well
            labels = []
            for i in range(self.total_events):
                # Get a row from the track dataframe
                track_param = self.targets.iloc[i].values.flatten().tolist()
                # Grab all target params up until the padded values
                if DIM == 2:
                    for i in range(0, len(track_param), DIM):
                        if track_param[i] == PAD_TOKEN:
                            break
                        labels.append(track_param[i])
                else: #dim==3
                    for i in range(0, len(track_param), DIM):
                        if track_param[i] == (PAD_TOKEN,PAD_TOKEN):
                            break
                        labels.append((track_param[i], track_param[i+1]))
            self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a row and transform it into a TODO what do the rows look like and also for targets ???
        row = self.data[idx]
        data = torch.tensor(row).float()

        if self.train:
            # If training, i.e. dataset has targets, obtain the track parameters
            row = self.targets[idx]
            target = torch.tensor(row).float()
        
        return idx, data, target