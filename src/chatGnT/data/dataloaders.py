import torch
from torch.utils.data import DataLoader, TensorDataset

def make_dataloaders_st(tensor, seed=42, batch_size=32, split=0.85):
    x = tensor[:, :-1]
    y = tensor[:, 1:]

    n = x.size(0)
    torch.manual_seed(seed)
    perm = torch.randperm(n)

    split_idx = int(split * n)
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    train_data = TensorDataset(x[train_idx], y[train_idx])
    val_data = TensorDataset(x[val_idx], y[val_idx])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def make_dataloaders_mt(amt_tensor, ingred_tensor, seed=42, batch_size=32, split=0.85):
    x_amt = amt_tensor[:, :-1]
    y_amt = amt_tensor[:, 1:]

    x_ingred = ingred_tensor[:, :-1]
    y_ingred = ingred_tensor[:, 1:]

    n = x_amt.size(0)
    torch.manual_seed(seed)
    perm = torch.randperm(n)

    split_idx = int(split * n)
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    train_data = TensorDataset(
        x_amt[train_idx], x_ingred[train_idx],
        y_amt[train_idx], y_ingred[train_idx]
    )

    val_data = TensorDataset(
        x_amt[val_idx], x_ingred[val_idx],
        y_amt[val_idx], y_ingred[val_idx]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
