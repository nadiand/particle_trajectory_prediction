import torch
import numpy as np
import pandas as pd
import math
import tqdm
from torch.nn.functional import pad
from timeit import default_timer as timer

from dataset import HitsDataset 
from transformer import TransformerModel, EarthMoverLoss
from global_constants import *
from dataloader import get_dataloaders
from visualization import visualize_tracks

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def earth_mover_loss(y_pred, y_true):
    # https://github.com/titu1994/neural-image-assessment/blob/master/train_mobilenet.py#L49
    y_pred = y_pred / (torch.sum(y_pred, dim=-1, keepdim=True) + 1e-14)
    y_true = y_true / (torch.sum(y_true, dim=-1, keepdim=True) + 1e-14)
    # print(y_pred)
    cdf_ytrue = torch.cumsum(y_true, axis=-1)
    cdf_ypred = torch.cumsum(y_pred, axis=-1)
    # print(cdf_ypred)
    samplewise_emd = torch.sqrt(torch.mean(torch.square(torch.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return torch.mean(samplewise_emd)

def input_mask(data):
    '''
    Create the input mask and input padding mask for the transformer.
    '''
    sequence_len = data.shape[0]
    padding_vector = torch.full((sequence_len,), PAD_TOKEN)
    mask = torch.zeros((sequence_len, sequence_len), device=DEVICE).type(torch.bool)
    padding_mask = (data.transpose(0, 2) == padding_vector).all(dim=0)
    return mask, padding_mask


def prediction_mask(preds, indices):
    '''
    Create the prediction mask, which masks all predictions for padded coordinates.
    '''
    indices_arr = np.array(indices)
    mask = torch.ones(preds.shape)
    for i, length in enumerate(indices_arr):
        mask[i][int(length):] = False
    return mask


def prep_labels(labels):
    '''
    Pad the labels to the maximum amount there can be (MAX_NR_TRACKS) so that they
    can be comared to the predictions, and mask the padded values.
    '''
    labels = labels.to(DEVICE)
    labels = pad(labels, (0,(MAX_NR_TRACKS-labels.shape[1])), "constant", PAD_TOKEN)
    label_mask = (labels != PAD_TOKEN).float()
    labels = labels * label_mask
    return labels


def make_prediction(model, data, real_lens):
    data = data.to(DEVICE)
    data = data.transpose(0,1) # TODO might be different for 3d
    padding_len = np.round(np.divide(real_lens, NR_DETECTORS))
    mask, padding_mask = input_mask(data)
    pred = model(data, mask, padding_mask)

    if DIM == 2:
        pred_mask = prediction_mask(pred, padding_len)
        pred = pred * torch.tensor(pred_mask).float()
    else: #dim==3
        # TODO try this out
        pred = pred[0].transpose(0, 1), pred[1].transpose(0, 1)
        pred = torch.stack([pred[0], pred[1]])
        for i in range(pred.shape[0]):
            slice_mask = prediction_mask(pred[i, :, :], padding_len)
            pred[i, :, :] = pred[i, :, :] * torch.tensor(slice_mask).float()
        pred = pred.transpose(0, 2)
        pred = pred.transpose(1, 0)
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
        _, x, labels, _, real_lens = data

        optim.zero_grad()

        # Make prediction
        pred = make_prediction(model, x, real_lens)
        # Pad and mask labels
        labels = prep_labels(labels)
        # Calculate loss and use it to update weights
        loss = loss_fn(pred, labels)
        # loss = earth_mover_distance(labels, pred)
        # loss = earth_mover_loss(pred, labels)
        # loss = chamfer_distance(pred.detach().numpy(), labels.detach().numpy())
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
            _, x, labels, _, real_lens = data

            # Make prediction
            pred = make_prediction(model, x, real_lens)
            # Pad and mask labels
            labels = prep_labels(labels)
            # Calculate loss
            loss = loss_fn(pred, labels)

            # if i == 1:
            #     print(pred[0], labels[0])
            #     visualize_tracks(pred.detach().numpy()[0], "predicted")
            #     visualize_tracks(labels.detach().numpy()[0], "true")
            #     exit()

            # loss = earth_mover_distance(labels, pred)
            # loss = earth_mover_loss(pred, labels)
            # loss = chamfer_distance(pred.detach().numpy(), labels.detach().numpy())
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
        event_id, x, _, _, real_lens = data

        # Make a prediction and append it to the list
        pred = make_prediction(model, x, real_lens)
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
    }, "direct_transformer_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks, True)
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
    loss_fn = EarthMoverLoss() #torch.nn.MSELoss()  #torch.nn.KLDivLoss(reduction="batchmean")
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
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count)
            count += 1

        # If the model hasn't improved in a while, stop the training
        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break

    # Predict on the test data
    preds = predict(transformer, test_loader)
    print(preds)
    # print(train_losses)