import torch
import numpy as np
import pandas as pd
import math
import tqdm

from dataset import HitsDataset 
from model_structures.classifier_transformer import TransformerClassifier, AsymmetricMSELoss
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_prediction(model, data):
    '''
    Create the input mask and the attention padding mask based on *data* and
    pass them to the model for prediction.
    '''
    data = data.to(DEVICE)
    data = data.transpose(0, 1)
    mask = torch.zeros((data.shape[0], data.shape[0]), device=DEVICE).type(torch.bool)
    # Make padding mask which conveys whether an element is padding or not
    padding_mask = (data == PAD_TOKEN).all(dim=2).T
    pred = model(data, mask, padding_mask)
    pred = pred.transpose(0, 1)
    return pred


def calc_accuracy(preds, labels):
    '''
    Calculate the accuracy of the model based on only the valid (i.e.
    non-PAD_TOKEN) hits' classification.
    '''
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(preds, axis=1)
    mask = np.logical_not(np.any(labels == PAD_TOKEN, axis=1))
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    accuracy = np.mean(y_true_filtered == y_pred_filtered)
    return accuracy


def train_epoch(model, optim, train_loader, loss_fn):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses, accuracy = 0., 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for _, data in t:
        _, x, _, track_labels = data
        optim.zero_grad()

        # Make prediction
        pred = make_prediction(model, x)
        # Mask the predictions and labels to ignore padded values during loss calculation
        mask = np.logical_not(np.any(track_labels.detach().numpy()[0] == PAD_TOKEN, axis=1))
        y_pred_filtered = pred.detach().numpy()[0][mask]
        y_true_filtered = track_labels.detach().numpy()[0][mask]
        # Calculate loss and use it to update weights
        loss = loss_fn(torch.tensor(y_pred_filtered, requires_grad=True).float(), torch.tensor(y_true_filtered, requires_grad=True).float())
        loss.backward()
        optim.step()
        losses += loss.item()

        # Calculate the accuracy of predictions
        acc = calc_accuracy(pred.detach().numpy()[0], track_labels.numpy()[0])
        accuracy += acc/len(x)
        t.set_description("loss = %.8f, accuracy = %.8f" % (loss.item(), acc/len(x)))

    return losses / len(train_loader), accuracy / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses, accuracy = 0., 0.
    n_batches = int(math.ceil(len(validation_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(validation_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            _, x, _, track_labels = data

            # Make prediction
            pred = make_prediction(model, x)
            # Mask the predictions and labels to ignore padded values during loss calculation
            mask = np.logical_not(np.any(track_labels.detach().numpy()[0] == PAD_TOKEN, axis=1))
            y_pred_filtered = pred.detach().numpy()[0][mask]
            y_true_filtered = track_labels.detach().numpy()[0][mask]
            # Calculate loss and accuracy
            loss = loss_fn(torch.tensor(y_pred_filtered, requires_grad=True).float(), torch.tensor(y_true_filtered, requires_grad=True).float())
            acc = calc_accuracy(pred.detach().numpy()[0], track_labels.numpy()[0])
            losses += loss.item()
            accuracy += acc

    return losses / len(validation_loader), accuracy / len(validation_loader)


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
    for i, data in t:
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
    }, "best_models/transformer_classifier_d1_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks, shuffle=False, sort_data=True)

    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    # Transformer model
    transformer = TransformerClassifier(num_encoder_layers=CL_NUM_ENCODER_LAYERS,
                                     d_model=CL_D_MODEL,
                                     n_head=CL_N_HEAD,
                                     input_size=DIM,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=CL_DIM_FEEDFORWARD,
                                     dropout=CL_DROPOUT)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))
    loss_fn = AsymmetricMSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CL_LEARNING_RATE)

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss, train_accuracy = train_epoch(transformer, optimizer, train_loader, loss_fn)
        # Evaluate on validation data
        val_loss, val_accuracy = evaluate(transformer, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, "
               f"Val loss: {val_loss:.8f}, Train loss: {train_loss:.8f}, "
               f"Val acc: {val_accuracy:.8f}, Train acc: {train_accuracy:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count)
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

    # Predict on the test data
    preds = predict(transformer, test_loader)
    print(preds)