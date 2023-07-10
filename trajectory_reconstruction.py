import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import math
import tqdm

from groups_dataset import GroupsDataset
from rnn_model import RNNModel
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_angle(rnn, clusters):
    padded_clusters, lens = [], []
    for cluster in clusters:
        pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        lens.append(len(cluster))
        padding = [pad for i in range(NR_DETECTORS-len(cluster))]
        padding = torch.tensor(padding).float()
        padded_clusters.append(torch.cat((cluster,padding),0))
    padded_clusters = torch.stack(padded_clusters)
    pred = rnn(torch.tensor(padded_clusters).float(), torch.tensor(lens).int())
    return pred if DIM == 2 else torch.stack((pred[0],pred[1]),dim=1)


def train(rnn, optim, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    rnn.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for i, data in t:
        _, group, label = data
        optim.zero_grad()
        pred = predict_angle(rnn, group)
        loss = loss_fn(pred, label)
        print(pred, label, loss) # in label there are PAD_TOKENS so the loss is in the 1000s TODO
        loss.backward()
        optim.step()
        losses += loss.item()
        t.set_description("loss = %.8f" % loss.item())

    return losses / len(train_loader)


def evaluation(rnn, val_loader, loss_fn):
    rnn.eval()
    losses = 0.
    n_batches = int(math.ceil(len(val_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(val_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for i, data in t:
            _, group, label = data
            pred = predict_angle(rnn, group)
            loss = loss_fn(pred, label)
            losses += loss.item()
            t.set_description("loss = %.8f" % loss.item())

    return losses / len(val_loader)


def prediction(rnn, test_loader):
    rnn.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for i, data in t:
            event_id, group, label = data
            pred = predict_angle(rnn, group)
            predictions[event_id] = (pred,label)#TODO remove the label from this function
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
    }, "rnn_"+type)


if __name__ == '__main__':
    torch.manual_seed(37)  # for reproducibility

    # load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = GroupsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)

    rnn = RNNModel(DIM, HIDDEN_SIZE_RNN, OUTPUT_SIZE_RNN)
    optim = torch.optim.Adam(rnn.parameters(), lr=RNN_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train(rnn, optim, train_loader, loss_fn)
        val_loss = evaluation(rnn, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(rnn, optim, "best", val_losses, train_losses, epoch, count)
            count = 0
        else:
            save_model(rnn, optim, "last", val_losses, train_losses, epoch, count)
            count += 1

        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break
    
    # print(prediction(rnn, test_loader))


    # transformer = TransformerClassifier(num_encoder_layers=NUM_ENCODER_LAYERS,
    #                                  d_model=D_MODEL,
    #                                  n_head=N_HEAD,
    #                                  input_size=INPUT_SIZE,
    #                                  output_size=OUTPUT_SIZE,
    #                                  dim_feedforward=DIM_FEEDFORWARD)
    # transformer = transformer.to(DEVICE)
    # transformer.eval()
    # optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    
    # checkpoint = torch.load("transformer_encoder_best")
    # transformer.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch'] + 1
    # train_losses = checkpoint['train_losses']
    # val_losses = checkpoint['val_losses']
    # min_val_loss = min(val_losses)
    # count = checkpoint['count']

    # torch.set_grad_enabled(True)
    # rnn.train()
    # losses = 0.
    # n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    # t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=False)
    # for i, data in t:
    #     event_id, x, labels, track_labels, real_lens = data
    #     x = x.to(DEVICE)

    #     preds = make_prediction(transformer, x, real_lens)
    #     groups = {}
    #     for i, pred in enumerate(preds):
    #         class_id = pred.argmax()
    #         if class_id in groups.keys():
    #             indices = groups[class_id]
    #             groups[class_id] = indices.append(i)
    #         else:
    #             groups[class_id] = [i]

    #     data = []
    #     x = x.detach().numpy()
    #     for key in groups.keys():
    #         indices = groups[key]
    #         data.append([x[i] for i in indices])

    #     print(data)
    #     optim.zero_grad()
    #     pred = rnn(torch.tensor(np.array(data)).float())
    #     loss = loss_fn(pred, labels)
    #     loss.backward()  # compute gradients
    #     optim.step()  # backprop
    #     losses += loss.item()
    #     t.set_description("loss = %.8f" % loss.item())

