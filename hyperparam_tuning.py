from functools import partial
import os
import torch
import pandas as pd
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from dataloader import get_dataloaders
from transformer import TransformerModel, EarthMoverLoss
from dataset import HitsDataset
from global_constants import *
from train_predict_transformer import train_epoch, evaluate


# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hyper_tune(config):
    transformer = TransformerModel(num_encoder_layers=config["num_encoder_layers"],
                                     d_model=config["d_model"],
                                     n_head=config["n_head"],
                                     input_size=INPUT_SIZE,
                                     output_size=OUTPUT_SIZE,
                                     dim_feedforward=config["dim_feedforward"],
                                     dropout=config["dropout"])
    # transformer = TransformerModel(num_encoder_layers=NUM_ENCODER_LAYERS,
    #                                  d_model=D_MODEL,
    #                                  n_head=N_HEAD,
    #                                  input_size=INPUT_SIZE,
    #                                  output_size=OUTPUT_SIZE,
    #                                  dim_feedforward=DIM_FEEDFORWARD)
    transformer = transformer.to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["lr"])
    loss_fn = config["loss"]

    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        transformer.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    hits = pd.read_csv(HITS_DATA_PATH, header=None)
    tracks = pd.read_csv(TRACKS_DATA_PATH, header=None)
    dataset = HitsDataset(hits, True, tracks)
    train_loader, val_loader, _ = get_dataloaders(dataset, config)

    train_losses, val_losses = [], []
    disable = False

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_epoch(transformer, optimizer, disable, train_loader, loss_fn)
        val_loss = evaluate(transformer, disable, val_loader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, "
               f"Val loss: {val_loss:.8f}"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"val loss": val_loss},
            checkpoint=checkpoint,
        )
    print("Done!")

if __name__ == '__main__':
    torch.manual_seed(7)

    config = {
        "num_encoder_layers": tune.choice([2, 4, 8]),
        "d_model": tune.choice([16, 32, 64, 128]),
        "n_head": tune.choice([2, 4, 8]),
        "dim_feedforward": tune.choice([1, 2, 4]),
        "dropout": tune.uniform(0.05, 0.2),
        "lr": tune.loguniform(1e-4, 1e-1),
        "loss": tune.choice([torch.nn.L1Loss(), torch.nn.KLDivLoss(reduction="batchmean"), EarthMoverLoss()]),
        "batch_size": tune.choice([16, 32, 64, 128])
    }

    scheduler = ASHAScheduler(
        metric="val loss",
        mode="min",
        max_t=NUM_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(hyper_tune),
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=100,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("val loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val loss']}")
