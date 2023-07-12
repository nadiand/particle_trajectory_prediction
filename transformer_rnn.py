import torch
import pandas as pd
import numpy as np

from dataset import HitsDataset
from model_structures.classifier_transformer import TransformerClassifier
from training.train_classifier_transformer import make_prediction, calc_accuracy
from training.train_rnn import predict_angle
from model_structures.rnn_model import RNNModel
from global_constants import *
from dataloader import get_dataloaders
from visualization import visualize_tracks

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load the dataset
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    _, _, test_loader = get_dataloaders(dataset)

    # Load best saved trained transformer model
    transformer = TransformerClassifier(num_encoder_layers=CL_NUM_ENCODER_LAYERS,
                                     d_model=CL_D_MODEL,
                                     n_head=CL_N_HEAD,
                                     input_size=DIM,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=CL_DIM_FEEDFORWARD,
                                     dropout=CL_DROPOUT)
    transformer = transformer.to(DEVICE)
    checkpoint = torch.load("best_models/transformer_classifier_best")
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()

    # Load best saved trained RNN model
    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    rnn = rnn.to(DEVICE)
    checkpoint = torch.load("best_models/rnn_best")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    rnn.eval()

    # Predicting
    predictions = {}
    accuracies = []
    for data in test_loader:
        event_id, x, labels, track_labels = data
        # Group hits into clusters with transformer
        cluster_predictions = make_prediction(transformer, x)
        cluster_predictions = cluster_predictions.detach().numpy()[0]
        # Calculate the accuracy of the classifications
        accuracies.append(calc_accuracy(cluster_predictions, track_labels.numpy()[0]))

        # Get the actual cluster IDs and ignore the predictions for padded hits
        cluster_IDs = np.argmax(cluster_predictions, axis=1)
        mask = []
        for row in x[0]:
            mask.append(not PAD_TOKEN in row)
        cluster_IDs = cluster_IDs[mask]

        # Convert x into a list, ignoring the padded values
        x_list = []
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
        for cluster_predictions in groups.values():
            if len(cluster_predictions) > NR_DETECTORS:
                culled_groups.append(cluster_predictions[:NR_DETECTORS])
            else:
                culled_groups.append(cluster_predictions)

        # Regress trajectory parameters and visualize the predicted track
        pred = predict_angle(rnn, culled_groups)
        predictions[event_id] = pred
        
        labels = labels.detach().numpy()[0]
        labels = labels[labels != PAD_TOKEN]
        visualize_tracks(pred.detach().numpy(), "predicted")
        visualize_tracks(labels, "true")

    print(pred)

