from dataset import HitsDataset
from global_constants import *
from transformer import FittingTransformer
from dataset import custom_collate #TODO: better placement for this one and the 2 above
import pandas as pd
import os
import torch
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import DataLoader, random_split
import math
import tqdm

# training function (to be called per epoch)
def train_epoch(model, optim, disable_tqdm, batch_size):
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=disable_tqdm)
    for i, data in t:
        event_id, x, y, z, tracks, labels = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if z is not None:
            z = z.to(DEVICE)
        if labels is not None:
            labels = labels.to(DEVICE)

        # run model
        if z is not None:
            pred = model(x, y, z)
        else:
            pred = model(x, y)

        optim.zero_grad()

        # loss calculation
        loss = loss_fn(pred, labels)
        loss.backward()  # compute gradients

        t.set_description("loss = %.8f" % loss.item())

        optim.step()  # backprop
        losses += loss.item()

    return losses / len(train_loader)


# test function
def evaluate(model, disable_tqdm, batch_size):
    model.eval()
    losses = 0
    n_batches = int(math.ceil(len(valid_loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(valid_loader), total=n_batches, disable=disable_tqdm)

    with torch.no_grad():
        for i, data in t:
            event_id, x, y, z, tracks, labels = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            if z is not None:
                z = z.to(DEVICE)
            tracks = tracks.to(DEVICE)
            if labels is not None:
                labels = labels.to(DEVICE)

            # create masks

            # run model
            pred = model(x, y, z)

            loss = loss_fn(pred, labels)
            losses += loss.item()

    return losses / len(valid_loader)

if __name__ == '__main__':
    hits = pd.read_csv(HITS_DATA_PATH)
    tracks = pd.read_csv(TRACKS_DATA_PATH)
    dataset = HitsDataset(hits, tracks)

    # manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # split dataset into training and validation sets
    full_len = len(dataset)
    train_len = int(full_len * 0.8)
    val_len = full_len - train_len
    train_set, val_set, = random_split(dataset, [train_len, val_len],
                                       generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              num_workers=4, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                              num_workers=4, shuffle=False, collate_fn=custom_collate)

    torch.manual_seed(7)  # for reproducibility
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformer model
    transformer = FittingTransformer(num_encoder_layers=4,
                                     d_model=32,
                                     n_head=4,
                                     input_size=3,
                                     output_size=3,
                                     dim_feedforward=1)
    transformer = transformer.to(DEVICE)
    print(transformer)

    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    # loss and optimiser
    loss_fn = torch.nn.MSELoss()  # earth_mover_distance  # EMD loss
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

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

    for epoch in range(epoch, 10 + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, disable, BATCH_SIZE)
        end_time = timer()
        val_loss = evaluate(transformer, disable, BATCH_SIZE)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, "
               f"Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("Saving best model with val_loss: {}".format(val_loss))
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'count': count,
            }, "transformer_encoder_best")
            count = 0
        else:
            print("Saving last model with val_loss: {}".format(val_loss))
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'count': count,
            }, "transformer_encoder_last")
            count += 1

        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break