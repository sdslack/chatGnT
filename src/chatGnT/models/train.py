from chatGnT.config import CFG
import torch
import torch.nn as nn
import time
import math
import copy
from itertools import product
from chatGnT.models.transformer import TransformerModel_SingleTask, TransformerModel_MultiTask
from chatGnT.models import evaluate
from chatGnT.models.structure import mask_single_task_output_logits
from chatGnT.data import dataloaders
import json
import os

# st = single task, mt = multi task

def train_st(model, dataloader, device, pad_id, optimizer, criterion, epoch=None, log_interval=None, vocab=None):
    model.train()  # turn on the train mode
    total_loss = 0.
    epoch_total_loss = 0.
    start_time = time.time()
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device).transpose(0, 1)   # (seq_len, batch)
        y = y.to(device).transpose(0, 1)

        # padding mask
        pad_mask = (x == pad_id).transpose(0, 1)  # (batch, seq_len)

        # causal mask
        seq_len = x.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        # forward pass
        output = model(
            src=x,
            src_key_padding_mask=pad_mask,
            src_mask=src_mask)

        if vocab is not None:
            output = mask_single_task_output_logits(output, vocab)

        # get loss
        loss = criterion(output.view(-1, output.size(-1)), y.reshape(-1))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        epoch_total_loss += loss.item()
        is_log_step = (log_interval and batch % log_interval == 0 and batch > 0)
        is_last_batch = (batch == num_batches - 1)

        if is_log_step or is_last_batch:
            elapsed = time.time() - start_time

            window_size = log_interval if is_log_step else (batch % log_interval)
            if window_size == 0: window_size = 1  # for single-batch epochs
            #TODO: is this correct?

            avg_loss = total_loss / window_size
            print(
                f'Epoch {epoch} | Batch {batch} | LR {optimizer.param_groups[0]["lr"]:.6f} | '
                f'Loss {avg_loss:.4f} | PPL {math.exp(avg_loss):.2f} | Time {elapsed:.2f}s')
            total_loss = 0
            start_time = time.time()

    return epoch_total_loss / num_batches

def train_mt(model, dataloader, device, pad_id_amt, pad_id_ingred, optimizer, criterion_amt, criterion_ingred, epoch=None, log_interval=None):
    model.train()  # turn on the train mode
    total_loss = 0.
    epoch_total_loss = 0.
    epoch_gradient_magnitude = 0.
    start_time = time.time()
    num_batches = len(dataloader)


    for batch, (x_amt, x_ingred, y_amt, y_ingred) in enumerate(dataloader):
        x_amt = x_amt.to(device).transpose(0, 1)  # (seq_len, batch)
        x_ingred = x_ingred.to(device).transpose(0, 1)
        y_amt = y_amt.to(device).transpose(0, 1)
        y_ingred = y_ingred.to(device).transpose(0, 1)

        # padding mask
        # pad_mask_amt = (x_amt == pad_id_amt).transpose(0, 1)
        # pad_mask_ingred = (x_ingred == pad_id_ingred).transpose(0, 1)
        pad_mask = (
            (x_amt == pad_id_amt) |
            (x_ingred == pad_id_ingred)
        ).transpose(0, 1)

        # causal mask
        seq_len = x_amt.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        # forward
        output_amt, output_ingred = model(
            src_amt=x_amt,
            src_ingred=x_ingred,
            src_key_padding_mask=pad_mask,
            src_mask=src_mask)

        # get losses
        loss_amt = criterion_amt(output_amt.view(-1, output_amt.size(-1)), y_amt.reshape(-1))
        loss_ingred = criterion_ingred(output_ingred.view(-1, output_ingred.size(-1)), y_ingred.reshape(-1))
        loss = loss_amt + loss_ingred

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        #accumulate gradient magnitude
        epoch_gradient_magnitude += torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # tracking
        total_loss += loss.item()
        epoch_total_loss += loss.item()
        is_log_step = (log_interval and batch % log_interval == 0 and batch > 0)
        is_last_batch = (batch == num_batches - 1)

        if is_log_step or is_last_batch:
            elapsed = time.time() - start_time

            window_size = log_interval if is_log_step else (batch % log_interval)
            if window_size == 0: window_size = 1  # for single-batch epochs
            #TODO: is this correct?

            avg_loss = total_loss / window_size
            print(
                f'Epoch {epoch} | Batch {batch} | LR {optimizer.param_groups[0]["lr"]:.6f} | '
                f'Loss {avg_loss:.4f} | PPL {math.exp(avg_loss):.2f} | Time {elapsed:.2f}s')
            total_loss = 0
            start_time = time.time()

    return epoch_total_loss / num_batches, epoch_gradient_magnitude/num_batches

def build_model(config, device):

    if config["model_version"] == "multi_task":
        model = TransformerModel_MultiTask(
            ntoken_amt=config["ntoken_amt"],
            ntoken_ingred=config["ntoken_ingred"],
            ninp=config["ninp"],
            nhead=config["nhead"],
            nhid=config["nhid"],
            nlayers=config["nlayers"]).to(device)
            # Note warning:
            # UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is
            # False because encoder_layer.self_attn.batch_first was not True (use
            # # batch_first for better inference performance)
    elif config["model_version"] == "single_task":
        model = TransformerModel_SingleTask(
            ntoken=config["ntoken"],
            ninp=config["ninp"],
            nhead=config["nhead"],
            nhid=config["nhid"],
            nlayers=config["nlayers"]).to(device)

    return model

def build_optimizer(model, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    return optimizer

def build_scheduler(optimizer, config):
    scheduler_type = config.get("scheduler_type", "step")

    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 4)
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler_step_size", 1),
            gamma=config.get("scheduler_gamma", 0.95)
        )
    else:
        raise ValueError(
            f"Unsupported scheduler_type: {scheduler_type}. "
            "Use 'reduce_on_plateau' or 'step'."
        )

    return scheduler


def step_scheduler(scheduler, config, val_loss=None):
    scheduler_type = config.get("scheduler_type", "step")
    if scheduler_type == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def build_dataloaders(config, tensors):
    if config["model_version"] == "multi_task":
        train_loader, val_loader = dataloaders.make_dataloaders_mt(
            tensors["amt_tensor"],
            tensors["ingred_tensor"],
            seed=config.get("seed", 42),
            batch_size=config["batch_size"],
            split=config.get("split", 0.85)
        )
    elif config["model_version"] == "single_task":
        train_loader, val_loader = dataloaders.make_dataloaders_st(
            tensors["tensor"],
            seed=config.get("seed", 42),
            batch_size=config["batch_size"],
            split=config.get("split", 0.85)
        )
    else:
        raise ValueError(f"Unsupported model_version: {config['model_version']}")

    return train_loader, val_loader


def build_criteria(config):
    if config["model_version"] == "multi_task":
        criterion_amt = nn.CrossEntropyLoss(ignore_index=config["pad_id_amt"])
        criterion_ingred = nn.CrossEntropyLoss(ignore_index=config["pad_id_ingred"])
        return criterion_amt, criterion_ingred
    if config["model_version"] == "single_task":
        return nn.CrossEntropyLoss(ignore_index=config["pad_id"])

    raise ValueError(f"Unsupported model_version: {config['model_version']}")


def run_training(config, tensors, device):
    """
    Train one model from config without requiring notebook-side setup of
    dataloaders, optimizer, scheduler, or loss functions.
    """
    train_loader, val_loader = build_dataloaders(config, tensors)
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    if config["model_version"] == "multi_task":
        criterion_amt, criterion_ingred = build_criteria(config)
        best_model, train_losses, val_losses, gradient_magnitudes = train_model_mt(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            scheduler,
            criterion_amt,
            criterion_ingred,
            config
        )
        return {
            "config": copy.deepcopy(config),
            "best_model": best_model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "gradient_magnitudes": gradient_magnitudes,
            "best_val_loss": min(val_losses),
        }

    criterion = build_criteria(config)
    best_model, train_losses, val_losses = train_model_st(
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        criterion,
        config
    )
    return {
        "config": copy.deepcopy(config),
        "best_model": best_model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses),
    }


def plot_training_history(train_losses, val_losses, gradient_magnitudes=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.show()

    if gradient_magnitudes is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(gradient_magnitudes, label="Gradient Magnitudes")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Gradient Magnitudes Loss Over Time")
        plt.legend()
        plt.show()


def save_artifacts_mt(best_model, config, vocab_amt, vocab_ingred, train_losses, val_losses, save_dir=(CFG.outputs_dir / "models")):
    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    with open(os.path.join(save_dir, "metrics_mt.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": {
            "ntoken_amt": config["ntoken_amt"],
            "ntoken_ingred": config["ntoken_ingred"],
            "ninp": config["ninp"],
            "nhead": config["nhead"],
            "nhid": config["nhid"],
            "nlayers": config["nlayers"],
        },
        "vocab_amt": vocab_amt,
        "vocab_ingred": vocab_ingred
    }, f"{save_dir}/model_mt.pt")

def save_artifacts_st(best_model, config, vocab, train_losses, val_losses, save_dir=(CFG.outputs_dir / "models")):
    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    with open(os.path.join(save_dir, "metrics_st.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": {
            "ntoken": config["ntoken"],
            "ninp": config["ninp"],
            "nhead": config["nhead"],
            "nhid": config["nhid"],
            "nlayers": config["nlayers"],
        },
        "vocab": vocab,
    }, f"{save_dir}/model.pt")


def iter_search_configs(base_config, search_space):
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]

    for combination in product(*values):
        trial_config = copy.deepcopy(base_config)
        for key, value in zip(keys, combination):
            trial_config[key] = value
        yield trial_config


def _format_search_trial_label(trial_num, trial_config, search_space):
    label_parts = [f"trial={trial_num}"]
    for key in search_space:
        label_parts.append(f"{key}={trial_config[key]}")
    return " ".join(label_parts)

def run_hyperparameter_search(base_config, search_space, tensors, device):
    results = []
    best_result = None

    for trial_num, trial_config in enumerate(iter_search_configs(base_config, search_space), start=1):
        trial_label = _format_search_trial_label(trial_num, trial_config, search_space)
        print(f"\nStarting {trial_label}")

        trial_result = run_training(trial_config, tensors, device)
        trial_result["trial"] = trial_num

        results.append(trial_result)
        print(
            f"Finished trial {trial_num} | "
            f"best_val_loss={trial_result['best_val_loss']:.4f}"
        )

        if best_result is None or trial_result["best_val_loss"] < best_result["best_val_loss"]:
            best_result = trial_result

    return best_result, results

def train_model_mt(model, train_loader, val_loader, device, optimizer, scheduler, criterion_amt, criterion_ingred, config):

    # Initialize trackers
    train_losses = []
    val_losses = []
    gradient_magnitudes = []
    best_val_loss = float("inf")
    best_model = None

    # Epochs & early stopping
    epochs = config["epochs"]  # number of epochs
    patience = config["early_stop"]  # Stop if no improvement for 5 epochs
    trigger_times = 0 

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        avg_train_loss, avg_gradient_magnitude = train_mt(
            model,
            train_loader,
            device,
            config["pad_id_amt"],
            config["pad_id_ingred"],
            optimizer,
            criterion_amt,
            criterion_ingred,
            epoch,
            config["log_interval"])
        train_losses.append(avg_train_loss)
        gradient_magnitudes.append(avg_gradient_magnitude)

        val_loss = evaluate.evaluate_mt(
            model,
            val_loader,
            device,
            config["pad_id_amt"],
            config["pad_id_ingred"],
            criterion_amt,
            criterion_ingred)
        val_losses.append(val_loss)

        print('-' * 89)
        print(
            f'Epoch {epoch} | Val Loss: {val_loss:.4f} | '
            f'Time {(time.time() - epoch_start_time)} | Val PPL: {math.exp(val_loss):.2f}')
        print('-' * 89)

        # Early stopping & best model check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model = copy.deepcopy(model)
        else:
            trigger_times += 1
            print(f'No improvement. Early stopping counter: {trigger_times}/{patience}')
            if trigger_times >= patience:
                print("Early stopping triggered. Ending training.")
                break

        step_scheduler(scheduler, config, val_loss)

    return best_model, train_losses, val_losses, gradient_magnitudes 

def train_model_st(model, train_loader, val_loader, device, optimizer, scheduler, criterion, config):

    # Initialize trackers
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = None

    # Epochs & early stopping
    epochs = config["epochs"]  # number of epochs
    patience = config["early_stop"]  # Stop if no improvement for 5 epochs
    trigger_times = 0 

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        avg_train_loss = train_st(
            model,
            train_loader,
            device,
            config["pad_id"],
            optimizer,
            criterion,
            epoch,
            config["log_interval"],
            config.get("vocab"))
        train_losses.append(avg_train_loss)

        val_loss = evaluate.evaluate_st(
            model,
            val_loader,
            device,
            config["pad_id"],
            criterion,
            config.get("vocab"))
        val_losses.append(val_loss)

        print('-' * 89)
        print(
            f'Epoch {epoch} | Val Loss: {val_loss:.4f} | '
            f'Time {(time.time() - epoch_start_time)} | Val PPL: {math.exp(val_loss):.2f}')
        print('-' * 89)

        # Early stopping & best model check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model = copy.deepcopy(model)
        else:
            trigger_times += 1
            print(f'No improvement. Early stopping counter: {trigger_times}/{patience}')
            if trigger_times >= patience:
                print("Early stopping triggered. Ending training.")
                break

        step_scheduler(scheduler, config, val_loss)

    return best_model, train_losses, val_losses
