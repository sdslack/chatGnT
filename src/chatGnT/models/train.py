import torch
import torch.nn as nn
import time
import math


def train(model, dataloader, device, pad_id, optimizer, criterion, epoch=None, log_interval=None):
    model.train()  # turn on the train mode
    total_loss = 0.
    epoch_total_loss = 0.
    start_time = time.time()
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)   # (batch, seq_len)
        y = y.to(device)
        x = x.transpose(0, 1)  # (seq_len, batch)
        y = y.transpose(0, 1)

        # padding mask
        pad_mask = (x == pad_id).transpose(0, 1)  # (batch, seq_len)

        # causal mask
        seq_len = x.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        # Forward pass
        output = model(
            src=x,
            src_key_padding_mask=pad_mask,
            src_mask=src_mask)

        vocab_size = output.size(-1)  # get number of classes
        loss = criterion(
            output.view(-1, vocab_size),
            y.reshape(-1))

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


def train_2head(model, dataloader, device, pad_id_amt, pad_id_ingred, optimizer, criterion_amt, criterion_ingred, epoch=None, log_interval=None):
    model.train()  # turn on the train mode
    total_loss = 0.
    epoch_total_loss = 0.
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

    return epoch_total_loss / num_batches

