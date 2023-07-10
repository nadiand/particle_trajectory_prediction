import pandas as pd
import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

from dataloader import get_dataloaders
from trajectory_reconstruction import RNNModel, predict_angle
from dataset import HitsDataset
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cluster_data(hits, visualize):
    '''
    Clusters hits based on the angles between them and returns the assignment of a cluster ID
    to each hit. If *visualize* is True, it also plots the identified clusters.
    '''
    print(len(hits))
    algo = AgglomerativeClustering(n_clusters=int(len(hits)/NR_DETECTORS), metric='cosine', linkage='average')

    # algo = AgglomerativeClustering(distance_threshold=0.004, n_clusters=None, metric='cosine', linkage='average')
    cls = algo.fit_predict(hits)

    if visualize:
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'brown', 'purple', 'black', 'cornflowerblue', 'pink', 'tomato', 'cyan', 'gray', 'orange', 'coral', 'seagreen', 'indigo', 'midnightblue', 'tan', 'lightcoral', 'gold', 'orchid', 'peachpuff', 'black', 'rosybrown', 'teal', 'plum', 'deeppink', 'purple']
        plt.figure()
        for i, point in enumerate(hits):
            plt.scatter(point[0], point[1], color=colors[cls[i]-1])
        plt.ylim([-6,6])
        plt.xlim([-6,6])
        plt.show()

    return cls

if __name__ == '__main__':
    # Prepare dataset
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    _, _, test_loader = get_dataloaders(dataset)

    # Load best trained RNN from file
    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    optim = torch.optim.Adam(rnn.parameters(), lr=TR_LEARNING_RATE)

    checkpoint = torch.load("rnn_best")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    min_val_loss = min(val_losses)
    count = checkpoint['count']

    # Predicting
    visualize = False
    predictions = {}
    for data in test_loader:
        event_id, x, labels, track_labels, _ = data
        # Convert x and track_labels into lists, ignoring the padded values
        tracks = track_labels[0].numpy()
        x_list, tracks_list = [], []
        for i, xx in enumerate(x[0]):
            if not PAD_TOKEN in xx:
                x_list.append(xx.tolist())
                tracks_list.append(tracks[i].argmax())
        
        # Cluster hits
        clusters = cluster_data(x_list, visualize)
        accuracy = accuracy_score(list(clusters), tracks_list)
        # TODO cull some points form clusters if you decide to do the threshold dist

        # Group hits together based on predicted cluster IDs
        groups = {}
        for i, lbl in enumerate(clusters):
            if lbl in groups.keys():
                groups[lbl].append(x_list[i])
            else:
                groups[lbl] = [x_list[i]]

        # Regress trajectory parameters
        pred = predict_angle(rnn, clusters)
        predictions[event_id] = pred
    
    print(predictions)