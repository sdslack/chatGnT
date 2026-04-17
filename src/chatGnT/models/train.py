import torch
import torch.nn as nn
import time
import math
import copy
from chatGnT.models.transformer import TransformerModel_SingleTask, TransformerModel_MultiTask
from chatGnT.models import evaluate

#TODO: re-org so evaluate not loaded here?
#TODO: switch train_st and train_mt to use config!
# st = single task, mt = multi task

def train_st(model, dataloader, device, pad_id, optimizer, criterion, epoch=None, log_interval=None):
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler_step_size"],
        gamma=config["scheduler_gamma"])
    return(scheduler)

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

        scheduler.step()  # adjusts learning rate

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
            config["log_interval"])
        train_losses.append(avg_train_loss)

        val_loss = evaluate.evaluate_st(
            model,
            val_loader,
            device,
            config["pad_id"],
            criterion)
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

        scheduler.step()  # adjusts learning rate

    return best_model, train_losses, val_losses

