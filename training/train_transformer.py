import torch
import numpy as np
import pandas as pd
import math
import tqdm

from dataset import HitsDataset 
from model_structures.transformer import TransformerModel, EarthMoverLoss
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_prediction(model, data):
    '''
    Transposes the data matrix and feed it into the model. Returns the prediction.
    '''
    data = data.to(DEVICE)
    data = data.transpose(0,1)
    pred = model(data)
    return pred


def train_epoch(model, optim, train_loader, loss_fn):
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

        optim.zero_grad()
        # Make prediction
        pred = make_prediction(model, x)
        # Calculate loss and use it to update weights
        loss = loss_fn(pred, labels)
        loss.backward() 
        optim.step()
        t.set_description("loss = %.8f" % loss.item())
        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0
    n_batches = int(math.ceil(len(validation_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(validation_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            _, x, labels, _ = data

            # Make prediction
            pred = make_prediction(model, x)
            # Calculate loss
            loss = loss_fn(pred, labels)
            losses += loss.item()

    return losses / len(validation_loader)


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

        # Make a prediction and append it to the list
        pred = make_prediction(model, x)
        for i, e_id in enumerate(event_id):
            predictions[e_id] = pred[i]

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
    }, "best_models/direct_transformer_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    # Transformer model
    transformer = TransformerModel(num_encoder_layers=TR_NUM_ENCODER_LAYERS,
                                     d_model=TR_D_MODEL,
                                     n_head=TR_N_HEAD,
                                     input_size=DIM,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=TR_DIM_FEEDFORWARD,
                                     dropout=TR_DROPOUT)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))
    loss_fn = EarthMoverLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=TR_LEARNING_RATE)

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    epoch, count = 0, 0

    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn)
        # Evaluate on validation data
        validation_loss = evaluate(transformer, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Val loss: {validation_loss:.8f}, Train loss: {train_loss:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(validation_loss)

        if validation_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = validation_loss
            save_model(transformer, optimizer, "best_new", val_losses, train_losses, epoch, count)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count)
            count += 1

        # If the model hasn't improved in a while, stop the training
        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

    # Predict on the test data
    preds = predict(transformer, test_loader)
    print(preds)