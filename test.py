import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dataset import HitsDataset
from dataloader import get_dataloaders
from regressor_model import RegressionModel
from trajectory_reconstruction import predict_angle
from rnn_model import RNNModel
from global_constants import *
from visualization import visualize_tracks
from clustering import cluster_data
import warnings
# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(threshold=100000000)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def load_regressor():
    # Regressor Model
    regressor = RegressionModel(DIM, HIDDEN_SIZE_REGRESS, OUTPUT_SIZE_REGRESS, DROPOUT_REGRESS)
    regressor = regressor.to(DEVICE)
    total_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(total_params))
    optimizer = torch.optim.Adam(regressor.parameters(), lr=LEARNING_RATE_REGRESS)
    loss_fn = torch.nn.MSELoss()

    checkpoint = torch.load("regressor_best")
    regressor.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plt.plot(train_losses,label='train')
    # plt.plot(val_losses,label='validation')
    # plt.legend()
    # plt.title("Loss evolution for D1 of best regressor model")
    # plt.show()
    return regressor, loss_fn

def load_rnn():
    # Load best trained RNN from file
    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    optim = torch.optim.Adam(rnn.parameters(), lr=RNN_LEARNING_RATE)
    total_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(total_params))
    loss_fn = torch.nn.MSELoss()

    checkpoint = torch.load("rnn_best_dataset1")
    rnn.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    min_val_loss = min(val_losses)
    count = checkpoint['count']

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='validation')
    plt.legend()
    plt.title("Loss evolution for D1 of best CL+RNN model")
    plt.show()
    return rnn, loss_fn

def use_clustering(track_labels, x):
    tracks = track_labels[0].numpy()
    x_list, tracks_list = [], []
    for i, xx in enumerate(x[0]):
        if not PAD_TOKEN in xx:
            x_list.append(xx.tolist())
            tracks_list.append(tracks[i].argmax())
    
    # Cluster hits
    clusters = cluster_data(x_list, False)
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
    return culled_groups

def mc_predict():
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv('hits_dataframe_dataset1.csv', header=None)
    tracks = pd.read_csv('tracks_dataframe_dataset1.csv', header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    model, loss_fn = load_regressor()
    # model, loss_fn = load_rnn()

    mc_dropout_preds = np.empty((0, len(test_loader), MAX_NR_TRACKS))
    error = 0
    labels, pr = [], []
    for i in range(5):
        model.eval()
        enable_dropout(model)
        predictions = np.empty((0, MAX_NR_TRACKS))
        for data in test_loader:
            event_id, x, label, track_labels = data
            x = x.to(DEVICE)
            labels.append(label)
            culled_groups = use_clustering(track_labels, x)
            with torch.no_grad():
                preds = model(x)
                # preds = predict_angle(model, culled_groups)
                loss = loss_fn(preds, label)#[0]) #for rnn
                error += loss.item()
                pr.append(preds)
                predictions = np.vstack((predictions, preds))
            # visualize_tracks(label[0].tolist(), "true")
            # visualize_tracks(preds.tolist(), "predicted by CL+RNN")
            # error = error + (labels[0] - preds[0])
        mc_dropout_preds = np.vstack((mc_dropout_preds, predictions[np.newaxis, :, :]))
    print(error/len(test_loader))
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(mc_dropout_preds, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(mc_dropout_preds, axis=0)  # shape (n_samples, n_classes)

    return mean, variance, labels, pr

if __name__ == '__main__':
    mc_predict()
    

    


        
