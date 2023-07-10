import torch
import numpy as np
from torch.utils.data import Dataset

from global_constants import *

class GroupsDataset(Dataset):
    def __init__(self, data, train, labels=None):
        self.data = data.fillna(value=PAD_TOKEN)
        self.train = train
        self.targets = labels
        if self.train:
            self.targets = self.targets.fillna(value=PAD_TOKEN)
        self.total_events = self.__len__()
        self.split_events()

    def split_events(self):
        split_data = []
        for i in range(self.total_events):
            row = self.data.iloc[[i]].values.tolist()[0]
            for j in range(0, len(row), NR_DETECTORS*(DIM+1)):
                if row[j] == PAD_TOKEN:
                    break
                x = row[j:j+NR_DETECTORS*(DIM+1):DIM+1]
                y = row[j+1:j+NR_DETECTORS*(DIM+1)+1:DIM+1]
                z = row[j+2:j+NR_DETECTORS*(DIM+1)+2:DIM+1] if DIM == 3 else None
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
            labels = []
            for i in range(self.total_events):
                event = self.targets.iloc[i].values.flatten().tolist()
                if DIM == 2:
                    for i in range(0, len(event), DIM):
                        if event[i] == PAD_TOKEN:
                            break
                        labels.append(event[i])
                else: #dim==3
                    for i in range(0, len(event), DIM):
                        if event[i] == (PAD_TOKEN,PAD_TOKEN):
                            break
                        labels.append((event[i], event[i+1]))
            
            self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        data = torch.tensor(row).float()

        if self.train:
            # If training, i.e. dataset has targets, obtain the track parameters
            row = self.targets[idx]
            target = torch.tensor(row).float()
        
        return idx, data, target