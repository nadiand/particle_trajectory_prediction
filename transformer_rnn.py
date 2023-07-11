import torch
import pandas as pd
import math
import tqdm
import numpy as np

from dataset import HitsDataset
from classifier_transformer import TransformerClassifier
from train_classifier_transformer import make_prediction, calc_accuracy
from trajectory_reconstruction import predict_angle
from rnn_model import RNNModel
from global_constants import *
from dataloader import get_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    _, _, test_loader = get_dataloaders(dataset)

    transformer = TransformerClassifier(num_encoder_layers=CL_NUM_ENCODER_LAYERS,
                                     d_model=CL_D_MODEL,
                                     n_head=CL_N_HEAD,
                                     input_size=DIM,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=CL_DIM_FEEDFORWARD,
                                     dropout=CL_DROPOUT)
    transformer = transformer.to(DEVICE)
    
    checkpoint = torch.load("transformer_classifier_best")
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()

    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    rnn = rnn.to(DEVICE)

    checkpoint = torch.load("rnn_best")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    rnn.eval()
    # rnn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    predictions = {}
    accuracies = []
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=False)
    for i, data in t:
        event_id, x, labels, track_labels = data
        group = make_prediction(transformer, x)
        group = group.detach().numpy()[0]
        accuracies.append(calc_accuracy(group, track_labels.numpy()[0]))

        cluster_IDs = np.argmax(group, axis=1)
        mask = []
        for row in x[0]:
            mask.append(not PAD_TOKEN in row)
        cluster_IDs = cluster_IDs[mask]

        x_list, tracks_list = [], []
        for i, xx in enumerate(x[0]):
            if not PAD_TOKEN in xx:
                x_list.append(xx.tolist())

        # Group hits together based on predicted cluster IDs
        groups = {}
        for i, lbl in enumerate(cluster_IDs):
            if lbl not in groups.keys():
                groups[lbl] = [x_list[i]]
            else:
                groups[lbl].append(x_list[i])
        
        # Prune the clusters so that they have at most NR_DETECTOR many hits
        culled_groups = []
        for group in groups.values():
            if len(group) > NR_DETECTORS:
                culled_groups.append(group[:NR_DETECTORS])
            else:
                culled_groups.append(group)

        # feed 1 friup at a time
        pred = predict_angle(rnn, culled_groups)
        predictions[event_id] = (pred,labels)

    avg_acc = sum(accuracies)/len(accuracies)
    print("The accuracy of the transformer per group:")
    print(accuracies)
    print(f"Average accuracy achieved by the transformer: {avg_acc}")
    print("The RNN's predicted track params:")
    print(predictions)
