import torch.nn as nn
import torch
import pandas as pd
import math
import numpy as np
import tqdm
from timeit import default_timer as timer

from dataset import HitsDataset
from dataloader import get_dataloaders
from regressor_model import RegressionModel
from regressor_constants import *
from global_constants import HITS_DATA_PATH, TRACKS_DATA_PATH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=False)

    for i, data in t:
        event_id, x, labels, _, _ = data
        x = x.to(DEVICE)

        outputs = model(x)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, val_loader, loss_fn):
    model.eval()
    losses = 0.
    n_batches = int(math.ceil(len(val_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(val_loader), total=n_batches, disable=False)

    for i, data in t:
        event_id, x, labels, _, _ = data
        x = x.to(DEVICE)
        preds = model(x)
        preds, _ = torch.sort(preds)
        # shapes and shit arent right, output is bad everything is bad TODO
        loss = loss_fn(preds, labels)
        losses += loss.item()

    return losses / len(val_loader)


def predict(model, test_loader):
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=False)

    for i, data in t:
        event_id, x, _, _, _ = data
        x = x.to(DEVICE)
        preds = model(x)
        preds, _ = torch.sort(preds)
        # shapes and shit arent right, output is bad everything is bad TODO
        predictions[event_id] = preds

    return predictions

if __name__ == '__main__':
    torch.manual_seed(7)  # for reproducibility

    regressor = RegressionModel(INPUT_SIZE_REGRESS, HIDDEN_SIZE_REGRESS, OUTPUT_SIZE_REGRESS, DROPOUT_REGRESS)
    regressor = regressor.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(regressor.parameters(), lr=LEARNING_RATE_REGRESS)

    # load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    min_val_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        start_time = timer()
        train_loss = train(regressor, train_loader, loss_fn)
        end_time = timer()
        validation_loss = evaluate(regressor, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Epoch time = {(end_time - start_time):.3f}s, "
               f"Val loss: {validation_loss:.8f}, Train loss: {train_loss:.8f}"))

        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            count = 0
        else:
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

    print(predict(regressor, test_loader))