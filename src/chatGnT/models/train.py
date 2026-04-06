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
