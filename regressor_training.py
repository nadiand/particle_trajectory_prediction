import torch
import numpy as np
import pandas as pd
import math
import tqdm

from dataset import HitsDataset
from dataloader import get_dataloaders
from regressor_model import RegressionModel
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, loss_fn):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for _, data in t:
        _, x, labels, _ = data
        x = x.to(DEVICE)

        # Make prediction
        outputs = model(x)
        # Calculate loss and use it to update the weights
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, val_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0.
    n_batches = int(math.ceil(len(val_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(val_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            _, x, labels, _ = data
            x = x.to(DEVICE)
            # Make prediction
            preds = model(x)
            # Calculate loss
            loss = loss_fn(preds, labels)
            losses += loss.item()

    return losses / len(val_loader)


def predict(model, test_loader):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=DISABLE_TQDM)

    for _, data in t:
        event_id, x, _, _ = data
        x = x.to(DEVICE)
        # Make a prediction and append it to the list
        preds = model(x)
        predictions[event_id] = preds

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
    }, "regressor_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv("hits_dataframe_dataset1.csv", header=None)
    tracks = pd.read_csv("tracks_dataframe_dataset1.csv", header=None)
    dataset = HitsDataset(hits, True, tracks, sort_targets=True)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    # Regressor Model
    regressor = RegressionModel(DIM, HIDDEN_SIZE_REGRESS, OUTPUT_SIZE_REGRESS, DROPOUT_REGRESS)
    regressor = regressor.to(DEVICE)
    total_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(total_params))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=LEARNING_RATE_REGRESS)

    # Training
    min_val_loss = np.inf
    train_losses, val_losses = [], []
    count = 0
    for epoch in range(500):#NUM_EPOCHS):
        # Train the model
        train_loss = train(regressor, train_loader, loss_fn)
        # Evaluate on validation data
        validation_loss = evaluate(regressor, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Val loss: {validation_loss:.8f}, Train loss: {train_loss:.8f}"))

        val_losses.append(validation_loss)
        train_losses.append(train_loss)

        if validation_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = validation_loss
            save_model(regressor, optimizer, "best", val_losses, train_losses, epoch, count)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(regressor, optimizer, "last", val_losses, train_losses, epoch, count)
            count += 1

        # If the model hasn't improved in a while, stop the training
        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break

    # Predict on the test data
    preds = predict(regressor, test_loader)
    # print(preds)