import torch
from torch.utils.data import DataLoader, random_split

from global_constants import *


def get_dataloaders(dataset):
    '''
    Split *dataset* into 3 subsets, train, validation and test, for the training
    and evaluating of a model.
    '''
    train_and_val = int(len(dataset) * (1-TEST_SPLIT))
    train_len = int(train_and_val * TRAIN_SPLIT)
    train_set, val_set, test_set = random_split(dataset, [train_len, len(dataset)-train_and_val, train_and_val-train_len], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader, test_loader


def get_dataloader_predicting(dataset):
    '''
    Generate a dataloader out of the dataset and return it. To be used for applying
    the model in practice.
    '''
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader