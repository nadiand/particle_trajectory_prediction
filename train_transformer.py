import pandas as pd
import os
import torch
import numpy as np
from timeit import default_timer as timer
import math
import tqdm

from dataset import HitsDataset 
from transformer import TransformerModel, EarthMoverLoss
from global_constants import *
from dataloader import get_dataloaders
from visualization import visualize_tracks

import pickle

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def chamfer_distance(y_true, y_pred):
    # formula from: https://proceedings.neurips.cc/paper_files/paper/2019/file/6e79ed05baec2754e25b4eac73a332d2-Paper.pdf
    # a set predictoin loss, not caring about placement
    distances = np.abs(y_pred[:, np.newaxis] - y_true)  # Calculate differences between each pair of pred and target
    min_dist = np.min(distances)
    min_dist = np.square(min_dist)
    return torch.tensor(min_dist, requires_grad=True)

def input_mask(data):
    src_seq_len = data.shape[0]
    padding_vector = torch.full((src_seq_len,), PAD_TOKEN)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (data.transpose(0, 2) == padding_vector).all(dim=0)
    return src_mask, src_padding_mask


def prediction_mask(preds, indices):
    indices_arr = np.array(indices)
    row_indices = np.arange(preds.shape[1])[:, np.newaxis]
    col_indices = np.arange(preds.shape[0])
    # if len(indices_arr) == 1:
    # print(indices_arr, row_indices, col_indices)
    mask = col_indices < indices_arr[row_indices]
    return mask.T


def prep_labels(labels):
    labels = labels.to(DEVICE)
    # Make label mask: by setting all pad tokans to 0 so they don't contribute to loss
    # TODO what happens actually? are they 0s and preds are 0s and we have no distance?
    # or is it 0s for labels and 101 for preds?? if the first thing, then thats bad!
    label_mask = (labels != PAD_TOKEN).float()
    labels = labels * label_mask
    return labels


def make_prediction(model, data, real_lens):
    data = data.transpose(0,1) #bc i dont use the collate function anymore
    padding_len = np.round(np.divide(real_lens, NR_DETECTORS))
    mask, padding_mask = input_mask(data)
    pred = model(data, mask, padding_mask)

    if DIM == 2:
        # pred = pred.transpose(0, 1)
        pred, _ = torch.sort(pred)
        pred_mask = prediction_mask(pred, padding_len)
        pred = pred * torch.tensor(pred_mask).float()
    else: #dim==3
        pred = pred[0].transpose(0, 1), pred[1].transpose(0, 1)
        # TODO sorting !!
        pred = torch.stack([pred[0], pred[1]])
        for slice_ind in range(pred.shape[0]):
            slice_mask = prediction_mask(pred[slice_ind, :, :], padding_len)
            pred[slice_ind, :, :] = pred[slice_ind, :, :] * torch.tensor(slice_mask).float()
        pred = pred.transpose(0, 2)
        pred = pred.transpose(1, 0)
    return pred


def train_epoch(model, optim, disable_tqdm, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=disable_tqdm)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x = x.to(DEVICE) #move to make_prediction TODO

        optim.zero_grad()

        labels = prep_labels(labels)
        pred = make_prediction(model, x, real_lens)
        loss = loss_fn(pred, labels)
        # loss = earth_mover_distance(labels, pred)
        # loss = earth_mover_loss(pred, labels)
        # loss = chamfer_distance(pred.detach().numpy(), labels.detach().numpy())
        loss.backward()  # compute gradients
        optim.step()  # backprop
        t.set_description("loss = %.8f" % loss.item())
        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, disable_tqdm, validation_loader, loss_fn):
    model.eval()
    losses = 0
    n_batches = int(math.ceil(len(validation_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(validation_loader), total=n_batches, disable=disable_tqdm)
    with torch.no_grad():
        for i, data in t:
            event_id, x, labels, track_labels, real_lens = data
            x = x.to(DEVICE)
            labels = prep_labels(labels)
            pred = make_prediction(model, x, real_lens)

            # if i == 1:
            #     visualize_tracks(pred.detach().numpy()[0], "predicted")
                # visualize_tracks(labels.detach().numpy()[0], "true")

            loss = loss_fn(pred, labels)
            # loss = earth_mover_distance(labels, pred)
            # loss = earth_mover_loss(pred, labels)
            # loss = chamfer_distance(pred.detach().numpy(), labels.detach().numpy())
            losses += loss.item()

    return losses / len(validation_loader)


def predict(model, test_loader, disable_tqdm):
    torch.set_grad_enabled(True)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=disable_tqdm)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x = x.to(DEVICE)

        pred = make_prediction(model, x, real_lens)
        # Append predictions to the list
        for i, e_id in enumerate(event_id):
            predictions[e_id] = pred[:, i]

    return predictions


def save_model(type):
    print(f"Saving {type} model with val_loss: {val_loss}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'count': count,
    }, "transformer_encoder_"+type)


if __name__ == '__main__':
    torch.manual_seed(7)  # for reproducibility

    # load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    # Transformer model
    transformer = TransformerModel(num_encoder_layers=NUM_ENCODER_LAYERS,
                                     d_model=D_MODEL,
                                     n_head=N_HEAD,
                                     input_size=INPUT_SIZE,
                                     output_size=OUTPUT_SIZE,
                                     dim_feedforward=DIM_FEEDFORWARD)
    transformer = transformer.to(DEVICE)
    # print(transformer)

    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    # loss and optimiser
    loss_fn = torch.nn.L1Loss() #EarthMoverLoss() #torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    min_val_loss = np.inf
    disable, load = False, False
    epoch, count = 0, 0

    if load:
        print("Loading saved model...")
        checkpoint = torch.load("models/transformer_encoder_generic_last")
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        min_val_loss = min(val_losses)
        count = checkpoint['count']
        print(epoch, val_losses)
    else:
        print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        start_time = timer() # TODO remove all the unnecessary timers and prints
        train_loss = train_epoch(transformer, optimizer, disable, train_loader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, disable, valid_loader, loss_fn)
        print((f"Train loss: {train_loss:.8f}, "
               f"Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model("best")
            count = 0
        else:
            save_model("last")
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

    # preds = predict(transformer, test_loader, disable)
    # print(preds)
    # with open('saved_dictionary.pkl', 'wb') as f:
    #     pickle.dump(preds, f)