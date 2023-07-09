import scipy.cluster.hierarchy as hc
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering

from dataloader import get_dataloaders
from dataset import HitsDataset
from global_constants import *

if __name__ == '__main__':
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    # train_loader, valid_loader, test_loader = get_dataloaders(dataset)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    subset = []
    labels = []
    clusters =[]
    for data in train_loader:
        event_id, x, _, l, real_lens = data
        for xx in x[0]:
            if xx != [PAD_TOKEN, PAD_TOKEN]:
                subset.append(xx.tolist())
        l_a = l.numpy().argmax()
        labels.append(l_a)

        algo = AgglomerativeClustering(distance_threshold=0.004, n_clusters=None, metric='cosine', linkage='average')
        cls = algo.fit_predict(subset)
        # Z = hc.linkage(subset, method='complete', metric='cosine')
        # cls = hc.fcluster(Z, criterion='maxclust', t=int(real_lens/NR_DETECTORS)) #assumes that the detectors are perfect and there's no missing hits!
        # clusters.append(cls)

        # clusters.append([c for c in cls])

    # print(subset)
    # print(labels)
    # print(clusters)
    # exit()
    print(cls)
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'brown', 'purple', 'black', 'cornflowerblue', 'pink', 'tomato', 'cyan', 'gray', 'orange', 'coral', 'seagreen', 'indigo', 'midnightblue', 'tan', 'lightcoral', 'gold', 'orchid', 'peachpuff']
    plt.figure()
    for i, point in enumerate(subset):
        plt.scatter(point[0], point[1], color=colors[cls[i]-1])
    plt.ylim([-6,6])
    plt.xlim([-6,6])
    plt.show()
