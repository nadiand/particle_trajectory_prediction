import pandas as pd
import os
import torch
import numpy as np
from timeit import default_timer as timer
import math
import tqdm

from dataset import HitsDataset 
from transformer import TransformerModel#, EarthMoverLoss
from global_constants import *
from dataloader import get_dataloaders

import pickle

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def EMD_loss(prediction, target):
#     # https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss/blob/master/pytorch_stats_loss.py
#     # normalize distribution, add 1e-14 to divisor to avoid 0/0
#     # this appears to work-ish without soring of the labels and preds, ie train loss goes down
#     # but the val loss doesnt... it gets worse
#     tensor_a = prediction / (torch.sum(prediction, dim=-1, keepdim=True) + 1e-14)
#     tensor_b = target / (torch.sum(target, dim=-1, keepdim=True) + 1e-14)
#     # make cdf with cumsum
#     cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
#     cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)
#     cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)), dim=-1)
#     cdf_loss = cdf_distance.mean()
#     return cdf_loss


def create_mask(data):
    src_seq_len = data.shape[0]
    padding_vector = torch.full((src_seq_len,), PAD_TOKEN)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (data.transpose(0, 2) == padding_vector).all(dim=0)
    return src_mask, src_padding_mask


def create_output_pred_mask(preds, indices):
    indices_arr = np.array(indices)
    row_indices = np.arange(preds.shape[1])[:, np.newaxis]
    col_indices = np.arange(preds.shape[0])
    mask = col_indices < indices_arr[row_indices]
    return mask.T


def prep_labels(labels):
    labels = labels.to(DEVICE)
    mask = (labels != PAD_TOKEN).float()
    labels = labels * mask
    return labels
    

def calc_loss(labels, pred, padding_len, loss_fn):
    pred_packed = torch.nn.utils.rnn.pack_padded_sequence(pred, padding_len, batch_first=False, enforce_sorted=False)
    tgt_packed = torch.nn.utils.rnn.pack_padded_sequence(labels, padding_len, batch_first=False, enforce_sorted=False)
    # TODO those above are essentially just transposes ! do those 

    # loss calculation
    # loss = EMD_loss(pred_packed.data, tgt_packed.data)
    loss = loss_fn(pred_packed.data, tgt_packed.data)
    return loss


def make_prediction(model, data, real_lens):
    padding_len = np.round(np.divide(real_lens, NR_DETECTORS))
    src_mask, src_padding_mask = create_mask(data)
    pred = model(data, src_mask, src_padding_mask)
    pred = pred.transpose(0, 1)

    pred, _ = torch.sort(pred)
    pred_mask = create_output_pred_mask(pred, padding_len)
    pred = pred * torch.tensor(pred_mask).float()
    return pred, padding_len


def train_epoch(model, optim, disable_tqdm, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=disable_tqdm)
    for i, data in t:
        event_id, x, real_lens, labels = data
        x = x.to(DEVICE)

        optim.zero_grad()

        labels = prep_labels(labels)
        pred, padding_len = make_prediction(model, x, real_lens)
        loss = calc_loss(labels, pred, padding_len, loss_fn)
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
            event_id, x, real_lens, labels = data
            x = x.to(DEVICE)
            labels = prep_labels(labels)
            pred, padding_len = make_prediction(model, x, real_lens)
            loss = calc_loss(labels, pred, padding_len, loss_fn)
            losses += loss.item()

    return losses / len(validation_loader)


def predict(model, test_loader, disable_tqdm):
    torch.set_grad_enabled(True)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=disable_tqdm)
    for i, data in t:
        event_id, x, real_lens, _ = data
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
    # load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    torch.manual_seed(7)  # for reproducibility

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
    loss_fn = torch.nn.L1Loss() #EarthMoverLoss()
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
        start_time = timer() # TODO remove all the unnecessary timers and prints
        train_loss = train_epoch(transformer, optimizer, disable, train_loader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, disable, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, "
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