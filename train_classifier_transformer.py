import torch
import numpy as np
import pandas as pd
import math
import tqdm
from sklearn.metrics import accuracy_score

from dataset import HitsDataset 
from classifier_transformer import TransformerClassifier
from global_constants import *
from dataloader import get_dataloaders
from visualization import visualize_tracks

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def input_mask(data):
#     src_seq_len = data.shape[0]
#     padding_vector = torch.full((src_seq_len,), PAD_TOKEN)
#     src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
#     src_padding_mask = (data.transpose(0, 2) == padding_vector).all(dim=0)
#     return src_mask, src_padding_mask


def prep_labels(labs):
    labs = labs.to(DEVICE)
    # Make label mask: by setting all pad tokens to 0
    label_mask = (labs != PAD_TOKEN).float()
    # print(label_mask)
    labs = labs * label_mask
    return labs


def make_prediction(model, data):
    data = data.to(DEVICE)
    data = data.transpose(0, 1)
    # move things to their dedicated function later on TODO
    mask = torch.zeros((data.shape[0], data.shape[0]), device=DEVICE).type(torch.bool)
    padding_mask = (data == PAD_TOKEN).all(dim=2).T
    pred = model(data, mask, padding_mask)
    pred = pred.transpose(0, 1)
    return pred


def calc_accuracy(preds, labels):
    y_true, y_pred = [], []
    for i, l in enumerate(labels):
        if not PAD_TOKEN in l:
            y_true.append(l.argmax())
            y_pred.append(preds[i].argmax())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def train_epoch(model, optim, train_loader, loss_fn):
    torch.set_grad_enabled(True)
    model.train()
    losses, accuracy = 0., 0.
    n_batches = int(math.ceil(len(train_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(train_loader), total=n_batches, disable=DISABLE_TQDM)
    for _, data in t:
        _, x, _, track_labels = data
        optim.zero_grad()
        pred = make_prediction(model, x)
        loss = loss_fn(pred, track_labels)
        loss.backward()
        optim.step()
        losses += loss.item()

        acc = calc_accuracy(pred.detach().numpy()[0], track_labels.numpy()[0])
        accuracy += acc/len(x)
        t.set_description("loss = %.8f, accuracy = %.8f" % (loss.item(), acc/len(x)))

    return losses / len(train_loader), accuracy / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    model.eval()
    losses, accuracy = 0., 0.
    n_batches = int(math.ceil(len(validation_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(validation_loader), total=n_batches, disable=DISABLE_TQDM)
    with torch.no_grad():
        for _, data in t:
            _, x, _, track_labels = data
            pred = make_prediction(model, x)

            # if i == 1:
            #     visualize_tracks(pred.detach().numpy()[0], "predicted")
                # visualize_tracks(labels.detach().numpy()[0], "true")

            loss = loss_fn(pred, track_labels)
            losses += loss.item()
            acc = calc_accuracy(pred.detach().numpy()[0], track_labels.numpy()[0])
            accuracy += acc/len(x)

    return losses / len(validation_loader), accuracy / len(validation_loader)


def predict(model, test_loader):
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    n_batches = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))
    t = tqdm.tqdm(enumerate(test_loader), total=n_batches, disable=DISABLE_TQDM)
    for i, data in t:
        event_id, x, _, _ = data

        pred = make_prediction(model, x)
        # Append predictions to the list
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
    }, "transformer_classifier_"+type)


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
                                     input_size=3,
                                     output_size=MAX_NR_TRACKS,
                                     dim_feedforward=CL_DIM_FEEDFORWARD,
                                     dropout=CL_DROPOUT)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CL_LEARNING_RATE)

    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_epoch(transformer, optimizer, train_loader, loss_fn)
        val_loss, val_accuracy = evaluate(transformer, valid_loader, loss_fn)
        print((f"Epoch: {epoch}, "
               f"Val loss: {val_loss:.8f}, Train loss: {train_loss:.8f}, "
               f"Val acc: {val_accuracy:.8f}, Train acc: {train_accuracy:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count)
            count = 0
        else:
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count)
            count += 1

        # if count >= EARLY_STOPPING:
        #     print("Early stopping...")
        #     break

    preds = predict(transformer, test_loader)
    print(preds)