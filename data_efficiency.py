import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader

import evidential_deep_learning as edl
from polymernet.data import PolymerDataset
from polymernet.model import SingleTaskNet


def normalization(dataset):
    ys = np.array([data.y for data in dataset])
    return ys.mean(), ys.std()


# ROOT_DIR = "./data/thermal_conductivity"
ROOT_DIR = "./data/heat_capacity"

LEARNING_RATE = 1e-4
EPOCHS = 100
INNER_EPOCHS = 50
BATCH_SIZE = 16
N_FEATURE = 16
N_LAYERS = 4
N_HIDDEN = 2
PATIENCE = 5
USE_LOG = True
HAS_H = False
FORM_RING = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(type):
    return PolymerDataset.from_file(
        ROOT_DIR,
        type,
        split=0,
        total_split=10,
        form_ring=FORM_RING,
        has_H=HAS_H,
        log10=USE_LOG,
        size_limit=None,
    )


def train_one_epoch(
    model,
    dataset,
    optimizer: torch.optim.Optimizer,
    mean,
    std,
    batch_size,
    shuffle=True,
):
    model.train()
    loss_sum = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        data.y = (data.y - mean) / std
        data.y = data.y.unsqueeze(1)
        loss = edl.losses.EvidentialRegression(data.y, output, coeff=0.02)
        loss.backward()
        loss_sum += loss.item() * data.num_graphs
        optimizer.step()
    return loss_sum / len(dataset)


def validate(
    model,
    val_dataset,
    mean,
    std,
    batch_size,
    shuffle=True,
):
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    error = 0
    # poly_ids = []
    preds = []
    targets = []
    vars = []

    for data in loader:
        data = data.to(device)
        output = model(data)
        output = output.detach().cpu().numpy()
        mu, v, alpha, beta = np.split(output, 4, axis=-1)
        mu = output[:, 0]
        var = np.sqrt(beta / (v * (alpha - 1)))
        var = np.minimum(var, 1e3)[:, 0]

        pred = mu
        pred = pred * std + mean
        target = data.y.cpu().detach().numpy()
        error += np.abs(pred - target).sum().item()
        preds.append(pred)
        targets.append(target)
        vars.append(var)
        # poly_ids += data.poly_id

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    vars = np.concatenate(vars)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    mae = error / len(loader.dataset)
    return mae, preds, targets, rmse, r2, vars


def train(model, train_dataset, val_dataset, n_epochs, batch_size, opt_and_sched=None):
    mean, std = normalization(train_dataset)

    if opt_and_sched is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=PATIENCE, min_lr=1e-7
        )
    else:
        optimizer, scheduler = opt_and_sched

    mae = []
    rmse = []
    r2 = []
    for epoch in range(n_epochs):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_dataset, optimizer, mean, std, batch_size)
        val_error, preds, targets, val_rmse, val_r2, val_vars = validate(
            model, val_dataset, mean, std, batch_size
        )
        scheduler.step(val_error)
        mae.append(val_error)
        rmse.append(val_rmse)
        r2.append(val_r2)
        print(
            "Epoch: {:03d}, LR: {:8f}, Loss: {:.5f}, Val MAE: {:.5f}, "
            "Val RMSE: {:.5f}, Val R2: {:.5f}".format(
                epoch,
                lr,
                loss,
                val_error,
                val_rmse,
                val_r2,
            ),
            file=sys.stderr,
        )
    return mae, rmse, r2


def train_active(initial_model, train_ds, val_ds, randomize=False, top_k=200):
    """
    Train via "active learning": focus on samples with largest uncertainty first.
    radomize: use random order instead (for ablation study)
    """
    data_example = train_ds[0]
    active_model = SingleTaskNet(
        data_example.num_features, data_example.num_edge_features, N_FEATURE, N_LAYERS, N_HIDDEN
    ).to(device)

    mean, std = normalization(train_ds)
    # Generate initial uncertainties
    _, _, _, _, _, vars = validate(initial_model, train_ds, mean, std, BATCH_SIZE)

    # Select samples with largest variance
    ids = np.argsort(vars)
    if randomize:
        np.random.shuffle(ids)
    top = list(ids[-top_k:])
    compl = list(ids[:-top_k])

    # Take top k for training, measure uncertainty for the rest.
    ds = Subset(train_ds, top)
    eval_ds = Subset(train_ds, compl)

    optimizer = torch.optim.Adam(active_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=PATIENCE, min_lr=1e-7
    )

    mae = []
    while True:
        subset_mean, subset_std = normalization(ds)

        print("ds lens:", len(ds), len(eval_ds), file=sys.stderr)
        m, _, _ = train(
            active_model,
            ds,
            val_ds,
            INNER_EPOCHS,
            BATCH_SIZE,
            opt_and_sched=(optimizer, scheduler),
        )
        mae += m

        if len(eval_ds) == 0:
            break

        _, _, _, _, _, vars = validate(
            active_model, eval_ds, subset_mean, subset_std, batch_size=BATCH_SIZE
        )

        ids = np.argsort(vars)
        if randomize:
            np.random.shuffle(ids)
        top = list(ids[-top_k:])
        compl = list(ids[:-top_k])
        # Add top to dataset, keep the rest for the next uncertainty round
        next_ds = ConcatDataset(datasets=[ds, Subset(eval_ds, top)])
        eval_ds = Subset(eval_ds, compl)
        ds = next_ds

    return mae


def main():
    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")
    # pred_dataset: PolymerDataset = load_dataset("pred")

    data_example = train_dataset[0]

    model = SingleTaskNet(
        data_example.num_features, data_example.num_edge_features, N_FEATURE, N_LAYERS, N_HIDDEN
    ).to(device)

    train(model, train_dataset, val_dataset, EPOCHS, BATCH_SIZE)

    # ams = []
    # rms = []
    N = 5
    print(N)
    for _ in range(N):
        active_mae = train_active(model, train_dataset, val_dataset, False)
        print(active_mae)
        random_mae = train_active(model, train_dataset, val_dataset, True)
        print(random_mae)
        # ams.append(active_mae)
        # rms.append(random_mae)

    # ams = np.array(ams)
    # print(ams.shape)
    # rms = np.array(rms)

    # ams_avg = np.mean(ams, 0)
    # ams_std = np.std(ams, 0)
    # rms_avg = np.mean(rms, 0)
    # rms_std = np.std(rms, 0)
    # plt.plot(ams_avg, label="Active")
    # plt.plot(rms_avg, label="Random")
    # plt.fill_between(list(range(len(ams_avg))), ams_avg + ams_std, ams_avg - ams_std, alpha=0.2)
    # plt.fill_between(list(range(len(rms_avg))), rms_avg + rms_std, rms_avg - rms_std, alpha=0.2)
    # plt.legend(loc="upper right")
    # plt.title("MAE over Epochs")
    # plt.show()


if __name__ == "__main__":
    main()
