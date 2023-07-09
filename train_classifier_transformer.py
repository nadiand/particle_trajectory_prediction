import pandas as pd
from sklearn import metrics
import torch
import numpy as np
from timeit import default_timer as timer
import math
import tqdm
from torch.utils.data import DataLoader

from dataset import HitsDataset 
from classifier_transformer import TransformerClassifier, ClusteringLoss
from global_constants import *
from dataloader import get_dataloaders
from visualization import visualize_tracks

import pickle

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


def prep_labels(labs):
    labs = labs.to(DEVICE)
    # Make label mask: by setting all pad tokens to 0
    label_mask = (labs != PAD_TOKEN).float()
    # print(label_mask)
    labs = labs * label_mask
    return labs


def make_prediction(model, data, real_lens):
    padding_len = np.round(np.divide(real_lens, NR_DETECTORS)) #move this in dataset TODO or better yet collate
    data = data.transpose(0, 1) #this should be in collate...TODO
    # move things to their dedicated function later on TODO
    mask = torch.zeros((data.shape[0], data.shape[0]), device=DEVICE).type(torch.bool)
    padding_mask = (data == PAD_TOKEN).all(dim=2).T
    pred = model(data, mask, padding_mask)
    pred = pred.transpose(0, 1)
    return pred


def accuracy_score(preds, labels):
    # TODO maybe we want to calculate the number of actual points to divide by, not the whole dataset
    accurate_points = 0
    inaccurate_pads = 0
    for p, l in zip(preds, labels):
        for p_x, l_x in zip(p, l):
            prediction = p_x.argmax()
            if any(l_x == PAD_TOKEN):
                if prediction > 0:
                    inaccurate_pads += 1
            else:
                if prediction == l_x.argmax():
                    accurate_points += 1
    # % of correctly classified points and % of nonpoints given a class
    return accurate_points, inaccurate_pads


def train_epoch(model, optim, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    model.train()
    losses, accuracy, inaccuracy = 0., 0., 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for i, data in t:
        event_id, x, labels, track_labels, real_lens = data
        x = x.to(DEVICE) #move to make_prediction TODO

        optim.zero_grad()
        pred = make_prediction(model, x, real_lens)
        loss = loss_fn(pred.detach().numpy(), track_labels.detach().numpy())
        loss.backward()  # compute gradients
        optim.step()  # backprop
        losses += loss.item()

        acc, inacc = accuracy_score(pred.detach().numpy(), track_labels.numpy())
        accuracy += acc/len(x)
        inaccuracy += inacc/len(x)
        t.set_description("loss = %.8f, accuracy = %.8f" % (loss.item(), acc/len(x)))

    return losses / len(train_loader), accuracy / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    model.eval()
    losses, accuracy, inaccuracy = 0., 0., 0.
    n_batches = int(math.ceil(len(validation_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(validation_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for i, data in t:
            event_id, x, labels, track_labels, real_lens = data
            x = x.to(DEVICE)
            pred = make_prediction(model, x, real_lens)

            # if i == 1:
            #     visualize_tracks(pred.detach().numpy()[0], "predicted")
                # visualize_tracks(labels.detach().numpy()[0], "true")

            loss = loss_fn(pred.detach().numpy(), track_labels.detach().numpy())
            losses += loss.item()
            acc, inacc = accuracy_score(pred.detach().numpy(), track_labels.numpy())
            accuracy += acc/len(x)
            inaccuracy += inacc/len(x)

    return losses / len(validation_loader), accuracy / len(validation_loader)


def predict(model, test_loader):
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=DISABLE_TQDM)
    for i, data in t:
        event_id, x, _, _, real_lens = data
        x = x.to(DEVICE)

        pred = make_prediction(model, x, real_lens)
        # Append predictions to the list
        for i, e_id in enumerate(event_id):
            predictions[e_id] = pred[i]
    
    return predictions


def save_model(type):
    # print(f"Saving {type} model with val_loss: {val_loss}")
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
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Transformer model
    transformer = TransformerClassifier(num_encoder_layers=CL_NUM_ENCODER_LAYERS,
                                     d_model=CL_D_MODEL,
                                     n_head=CL_N_HEAD,
                                     input_size=DIM,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=CL_DIM_FEEDFORWARD)
    transformer = transformer.to(DEVICE)
    # print(transformer)

    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    # loss and optimiser
    loss_fn = ClusteringLoss() #torch.nn.MSELoss() #torch.nn.L1Loss() #torch.nn.KLDivLoss(reduction="batchmean") #EarthMoverLoss() 
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CL_LEARNING_RATE)

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
        train_loss, train_accuracy = train_epoch(transformer, optimizer, train_loader, loss_fn)
        end_time = timer()
        exit()
        val_loss, val_accuracy = evaluate(transformer, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, Epoch time = {(end_time - start_time):.3f}s, "
               f"Val loss: {val_loss:.8f}, Train loss: {train_loss:.8f}, "
               f"Val acc: {val_accuracy:.8f}, Train acc: {train_accuracy:.8f}"))

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

    preds = predict(transformer, test_loader)
    # print(preds[0])
    print(preds)
    # with open('saved_dictionary.pkl', 'wb') as f:
    #     pickle.dump(preds, f)