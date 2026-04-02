import torch
import torch.nn as nn
import time
import math

def train(model, dataloader, device, pad_id, optimizer, criterion, epoch=None, log_interval=None):
    model.train()  # turn on the train mode
    total_loss = 0.
    start_time = time.time()

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

        if log_interval and epoch and batch % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / log_interval
            print(
                f'Epoch {epoch} | Batch {batch} | LR {optimizer.param_groups[0]["lr"]:.6f} | '
                f'Loss {avg_loss:.4f} | PPL {math.exp(avg_loss):.2f} | Time {elapsed:.2f}s')
            total_loss = 0
            start_time = time.time()

