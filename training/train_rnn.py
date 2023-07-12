import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import math
import tqdm

from groups_dataset import GroupsDataset
from model_structures.rnn_model import RNNModel
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_angle(rnn, clusters):
    '''
    Regresses the trajectory parameters based on the clusters of hits. Takes a set of
    *clusters*, pads them up to 5 hits each if necessary, and passes them on to the 
    *rnn* model. Directly returns the predictions.
    '''
    padded_clusters, lens = [], []
    for cluster in clusters:
        # Keep track of real (unpadded) sizes of clusters, to pass on to the RNN
        lens.append(len(cluster))

        # Make padding based on how many hits each cluster is missing
        pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        padding = [pad for _ in range(NR_DETECTORS-len(cluster))]

        padding = torch.tensor(padding).float()
        if isinstance(cluster, list):
            cluster = torch.tensor(cluster).float()
        
        # Pad the clusters with pseudo hits
        padded_clusters.append(torch.cat((cluster,padding),0))
    padded_clusters = torch.stack(padded_clusters)

    pred = rnn(torch.tensor(padded_clusters).float(), torch.tensor(lens).int())
    return pred if DIM == 2 else torch.stack((pred[0],pred[1]),dim=1)


def train(rnn, optim, train_loader, loss_fn):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    rnn.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for _, data in t:
        _, group, label = data

        optim.zero_grad()
        # Make prediction
        pred = predict_angle(rnn, group)
        # Mask the predictions and labels to ignore padded values during loss calculation
        mask = [not PAD_TOKEN in l for l in label]
        pred = pred[mask]
        label = label[mask]
        # Calculate loss and use it to update weights
        loss = loss_fn(pred, label)
        loss.backward()
        optim.step()
        losses += loss.item()
        t.set_description("loss = %.8f" % loss.item())

    return losses / len(train_loader)


def evaluation(rnn, val_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    rnn.eval()
    losses = 0.
    n_batches = int(math.ceil(len(val_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(val_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            _, group, label = data

            # Make prediction
            pred = predict_angle(rnn, group)
            # Mask predictions and labels to ignore padded values during loss calculation
            mask = [not PAD_TOKEN in l for l in label]
            pred = pred[mask]
            label = label[mask]
            # Calculate loss
            loss = loss_fn(pred, label)
            losses += loss.item()
            t.set_description("loss = %.8f" % loss.item())

    return losses / len(val_loader)


def prediction(rnn, test_loader):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    rnn.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            event_id, group, label = data
            # Make prediction and add it to the list
            pred = predict_angle(rnn, group)
            predictions[event_id] = (pred,label)
    return predictions


def save_model(model, optim, type, val_losses, train_losses, epoch, count):
    print(f"Saving {type} model")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'count': count,
    }, "best_models/rnn_dataset3_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = GroupsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    # RNN model
    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    pytorch_total_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))
    optim = torch.optim.Adam(rnn.parameters(), lr=RNN_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0
    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss = train(rnn, optim, train_loader, loss_fn)
        # Evaluate on validation data
        val_loss = evaluation(rnn, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            save_model(rnn, optim, "best_new", val_losses, train_losses, epoch, count)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(rnn, optim, "last", val_losses, train_losses, epoch, count)
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break
    
    # Predict on the test data
    preds = prediction(rnn, test_loader)
    print(preds)
