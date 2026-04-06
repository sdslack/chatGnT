import torch

def evaluate(model, dataloader, device, pad_id, criterion):
    model.eval()  # turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            x = x.transpose(0, 1)  # (seq_len, batch)
            y = y.transpose(0, 1)

            # padding mask
            pad_mask = (x == pad_id).transpose(0, 1)

            # causal mask
            seq_len = x.size(0)
            src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

            # forward
            output = model(
                src=x,
                src_key_padding_mask=pad_mask,
                src_mask=src_mask)
            # output_flat = output.view(-1, ntokens)

            vocab_size = output.size(-1)  # get number of classes
            loss = criterion(
                output.view(-1, vocab_size),
                y.reshape(-1))
            
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

