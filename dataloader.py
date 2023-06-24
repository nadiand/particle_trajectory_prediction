import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from global_constants import *

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
        labels.append(sample[4])

    real_data_len = [len(val) for val in xs]
    xs_pad, ys_pad, zs_pad, lab_pad = pad(xs), pad(ys), pad(zs), pad(labels)
    x = torch.stack((xs_pad, ys_pad, zs_pad), dim=1)

    # Return the final processed batch
    return event_ids, x.transpose(1,2), real_data_len, lab_pad

def get_dataloaders(dataset):
    train_and_val = int(len(dataset) * (1-TEST_SPLIT))
    train_len = int(train_and_val * TRAIN_SPLIT)
    train_set_full, val_set, = random_split(dataset, [train_and_val, (len(dataset)-train_and_val)], generator=torch.Generator().manual_seed(7))
    train_set, test_set = random_split(train_set_full, [train_len, (train_and_val-train_len)], generator=torch.Generator().manual_seed(7))
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, num_workers=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader