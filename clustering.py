import scipy.cluster.hierarchy as hc
import pandas as pd
import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn
import torch

from dataloader import get_dataloaders
from trajectory_reconstruction import RNNModel
from dataset import HitsDataset
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cluster_data(hits):
    algo = AgglomerativeClustering(distance_threshold=0.004, n_clusters=None, metric='cosine', linkage='average')
    cls = algo.fit_predict(hits)
    if False:
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'brown', 'purple', 'black', 'cornflowerblue', 'pink', 'tomato', 'cyan', 'gray', 'orange', 'coral', 'seagreen', 'indigo', 'midnightblue', 'tan', 'lightcoral', 'gold', 'orchid', 'peachpuff', 'black', 'rosybrown', 'teal', 'plum', 'deeppink', 'purple']
        plt.figure()
        for i, point in enumerate(hits):
            plt.scatter(point[0], point[1], color=colors[cls[i]-1])
        plt.ylim([-6,6])
        plt.xlim([-6,6])
        plt.show()
    return cls

if __name__ == '__main__':
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rnn = RNNModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optim = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load("rnn_best")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    min_val_loss = min(val_losses)
    count = checkpoint['count']
    # loss_fn = nn.MSELoss()
    # torch.set_grad_enabled(True)
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / TEST_BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=False)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x_list = []
        for xx in x[0]:
            if xx != [PAD_TOKEN, PAD_TOKEN] and xx != [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]:
                x_list.append(xx.tolist())
        clusters = cluster_data(x_list)
        print(clusters, track_labels)
        groups = {}
        for i, lbl in enumerate(clusters):
            lbl = lbl.argmax()
            if lbl in groups.keys():
                groups[lbl].append(x_list[i])
            else:
                groups[lbl] = [x_list[i]]
        biggest_cl = max([len(x) for x in groups.values()])
        data, seq_lengths = [], []
        for key in groups.keys():
            length = len(groups[key])
            seq_lengths.append(length)
            pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
            padding = [pad for i in range(biggest_cl-length)]
            data.append(groups[key] + padding)

        pred = rnn(torch.tensor(data).float(), torch.tensor(seq_lengths).int())
        predictions[event_id] = pred
        print(pred, labels)

    print(predictions)
    # rnn.train()
    # losses = 0.
    # n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    # t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=False)
    # for i, data in t:
    #     event_id, x, labels, track_labels, real_lens = data
    #     x_list = []
    #     for xx in x[0]:
    #         if xx != [PAD_TOKEN, PAD_TOKEN]:
    #             x_list.append(xx.tolist())
    #     clusters = cluster_data(x_list)

    #     groups = {}
    #     for i, pred in enumerate(clusters):
    #         if pred in groups.keys():
    #             groups[pred].append(i)
    #         else:
    #             groups[pred] = [i]
    #     biggest_cl = max([len(x) for x in groups.values()])
    #     data = []
    #     seq_lengths = []
    #     for key in groups.keys():
    #         indices = groups[key]
    #         seq_lengths.append(len(indices))
    #         padding = [[PAD_TOKEN, PAD_TOKEN] for i in range(biggest_cl-len(indices))] #for 2d only
    #         data.append([x_list[i] for i in indices] + padding)

    #     optim.zero_grad()
    #     pred = rnn(torch.tensor(data).float(), torch.tensor(seq_lengths).int())
    #     loss = loss_fn(pred, labels[0])
    #     loss.backward()  # compute gradients
    #     optim.step()  # backprop
    #     losses += loss.item()
    #     t.set_description("loss = %.8f" % loss.item())

