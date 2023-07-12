import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch

from model_structures.rnn_model import RNNModel
from training.train_rnn import predict_angle
from dataset import HitsDataset
from visualization import visualize_tracks
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cluster_data(hits, visualize):
    '''
    Clusters hits based on the angles between them and returns the assignment of a cluster ID
    to each hit. If *visualize* is True, it also plots the identified clusters.
    '''
    algo = AgglomerativeClustering(n_clusters=int(len(hits)/NR_DETECTORS), metric='cosine', linkage='average')
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


def calculate_purity(clusters, tracks_list):
    '''
    Calculates the purity of the clusters based on the ground truth given
    by tracks_list.
    '''
    purity = 0.
    cluster_list = list(clusters)
    # Iterate over all clusters in the ground truth
    for class_lbl in range(max(tracks_list)):
        # For each cluster, get the indices of hits belonging to it
        indices = [i for i,x in enumerate(tracks_list) if x == class_lbl]
        # Get the corresponding predicted hits
        cluster_indexed = [cluster_list[i] for i in indices]
        # Calcualte the majority count
        majority_label = np.argmax(np.bincount(cluster_indexed))
        majority_count = np.sum(cluster_indexed == majority_label)
        purity += majority_count
    return purity/len(tracks_list)


if __name__ == '__main__':
    # Prepare dataset
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    _, _, test_loader = get_dataloaders(dataset)

    # Load best trained RNN from file
    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    optim = torch.optim.Adam(rnn.parameters(), lr=TR_LEARNING_RATE)

    checkpoint = torch.load("best_models/rnn_best")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    # Predicting
    visualize = False
    purity = 0.
    predictions = {}
    for data in test_loader:
        event_id, x, labels, track_labels = data
        # Convert x and track_labels into lists, ignoring the padded values
        tracks = track_labels[0].numpy()
        x_list, tracks_list = [], []
        for i, xx in enumerate(x[0]):
            if not PAD_TOKEN in xx:
                x_list.append(xx.tolist())
                tracks_list.append(tracks[i].argmax())
        
        # Cluster hits
        clusters = cluster_data(x_list, visualize)
        purity += calculate_purity(clusters, tracks_list)

        # Group hits together based on predicted cluster IDs
        groups = [ [] for _ in range(max(clusters)+1) ]
        for i, lbl in enumerate(clusters):
            groups[lbl].append(x_list[i])

        # Prune the clusters so that they have at most NR_DETECTOR many hits
        culled_groups = []
        for group in groups:
            if len(group) > NR_DETECTORS:
                culled_groups.append(group[:NR_DETECTORS])
            else:
                culled_groups.append(group)
        # Regress trajectory parameters and visualize the predicted track
        pred = predict_angle(rnn, culled_groups)
        print(purity/len(test_loader))
        predictions[event_id] = pred

        labels = labels.detach().numpy()[0]
        labels = labels[labels != PAD_TOKEN]
        visualize_tracks(pred.detach().numpy(), "predicted by CL+RNN")
        visualize_tracks(labels, "true")
    
    print(predictions)