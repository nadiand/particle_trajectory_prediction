import torch.nn as nn
import torch
import pandas as pd
from torch.autograd import Variable
import math
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from dataset import HitsDataset
from second_transformer import TransformerClassifier
from new_training import make_prediction
from dataloader import get_dataloaders
from global_constants import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, input, seq_lengths):
        seq_lengths, sorted_idx = seq_lengths.sort(descending=True)
        input = input[sorted_idx]
        batch_size = input.shape[0]

        padded = nn.utils.rnn.pack_padded_sequence(input, seq_lengths, batch_first=True)

        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))
        output, (final_hidden_state, final_cell_state) = self.lstm(padded, (h_0, c_0))
        out = self.fc(final_hidden_state[-1]) 
        out = torch.mean(out, dim=0)
        return out
    
def train(rnn, optim, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    rnn.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=False)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x_list = []
        for xx in x[0]:
            if xx != [PAD_TOKEN, PAD_TOKEN]:
                x_list.append(xx.tolist())
        groups = {}
        for i, lbl in enumerate(track_labels.numpy()[0]):
            lbl = lbl.argmax()
            if lbl in groups.keys():
                groups[lbl].append(x_list[i])
            else:
                groups[lbl] = [x_list[i]]
        biggest_cl = max([len(x) for x in groups.values()])
        data = []
        seq_lengths = []
        for key in groups.keys():
            length = len(groups[key])
            seq_lengths.append(length)
            pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
            padding = [pad for i in range(biggest_cl-length)]
            data.append(groups[key] + padding)
        optim.zero_grad()
        pred = rnn(torch.tensor(data).float(), torch.tensor(seq_lengths).int())
        loss = loss_fn(pred, labels[0])
        # print(pred, labels[0])
        loss.backward()  # compute gradients
        optim.step()  # backprop
        losses += loss.item()
        t.set_description("loss = %.8f" % loss.item())

    return losses / len(train_loader)

def evaluation(rnn, val_loader, loss_fn):
    rnn.eval()
    losses = 0.
    n_batches = int(math.ceil(len(val_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(val_loader), total=n_batches, disable=False)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x_list = []
        for xx in x[0]:
            if xx != [PAD_TOKEN, PAD_TOKEN]:
                x_list.append(xx.tolist())
        groups = {}
        for i, lbl in enumerate(track_labels.numpy()[0]):
            lbl = lbl.argmax()
            if lbl in groups.keys():
                groups[lbl].append(x_list[i])
            else:
                groups[lbl] = [x_list[i]]
        biggest_cl = max([len(x) for x in groups.values()])
        data = []
        seq_lengths = []
        for key in groups.keys():
            length = len(groups[key])
            seq_lengths.append(length)
            pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
            padding = [pad for i in range(biggest_cl-length)]
            data.append(groups[key] + padding)
        pred = rnn(torch.tensor(data).float(), torch.tensor(seq_lengths).int())
        loss = loss_fn(pred, labels[0])
        losses += loss.item()
        t.set_description("loss = %.8f" % loss.item())

    return losses / len(val_loader)


def prediction(rnn, test_loader):
    rnn.eval()
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=False)
    predictions = {}
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x_list = []
        for xx in x[0]:
            if xx != [PAD_TOKEN, PAD_TOKEN] and xx != [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]:
                x_list.append(xx.tolist())
        groups = {}
        for i, lbl in enumerate(track_labels.numpy()[0]):
            lbl = lbl.argmax()
            if lbl in groups.keys():
                groups[lbl].append(x_list[i])
            else:
                groups[lbl] = [x_list[i]]
        biggest_cl = max([len(x) for x in groups.values()])
        data, seq_lengths = [], []
        for key in groups.keys():
            length = len(groups[key])
            seq_lengths.append(length)
            pad = [PAD_TOKEN, PAD_TOKEN] if DIM == 2 else [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
            padding = [pad for i in range(biggest_cl-length)]
            data.append(groups[key] + padding)
        pred = rnn(torch.tensor(data).float(), torch.tensor(seq_lengths).int())
        predictions[event_id] = pred
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
    torch.manual_seed(7)  # for reproducibility

    # load and split dataset into training, validation and test sets
    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset)
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rnn = RNNModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optim = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        start_time = timer()
        train_loss = train(rnn, optim, train_loader, loss_fn)
        end_time = timer()
        val_loss = 0
        val_loss = evaluation(rnn, valid_loader, loss_fn)
        print((f"Train loss: {train_loss:.8f}, "
               f"Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

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
    
    print(prediction(rnn, test_loader))
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

