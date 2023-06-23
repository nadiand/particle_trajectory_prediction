import pandas as pd
import os
import torch
import numpy as np
from timeit import default_timer as timer
import math
import tqdm
from dataset import HitsDataset 
from transformer import FittingTransformer
from global_constants import *
from dataloader import get_dataloaders

# training function (to be called per epoch)
def train_epoch(model, optim, disable_tqdm):
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
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
def evaluate(model, disable_tqdm):
    model.eval()
    losses = 0
    n_batches = int(math.ceil(len(valid_loader.dataset) / BATCH_SIZE))
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
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)

    # manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # split dataset into training, validation and test sets
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)
    # train_and_val = int(len(dataset) * (1-TEST_SPLIT))
    # train_len = int(train_and_val * TRAIN_SPLIT)
    # train_set_full, val_set, = random_split(dataset, [train_and_val, (len(dataset)-train_and_val)], generator=torch.Generator().manual_seed(7))
    # train_set, test_set = random_split(train_set_full, [train_len, (train_and_val-train_len)], generator=torch.Generator().manual_seed(7))
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collate_fn)
    # valid_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(7)  # for reproducibility
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformer model
    transformer = FittingTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
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
    loss_fn = torch.nn.MSELoss() 
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

    for epoch in range(NUM_EPOCHS):
        start_time = timer() # TODO remove all the unnecessary timers and prints
        train_loss = train_epoch(transformer, optimizer, disable)
        end_time = timer()
        val_loss = evaluate(transformer, disable)
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

        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break