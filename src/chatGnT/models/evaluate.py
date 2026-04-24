import torch
from chatGnT.models.structure import mask_single_task_output_logits

# st = single task, mt = multi task

def evaluate_st(model, dataloader, device, pad_id, criterion, vocab=None):
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

            if vocab is not None:
                output = mask_single_task_output_logits(output, vocab)

            vocab_size = output.size(-1)  # get number of classes
            loss = criterion(
                output.view(-1, vocab_size),
                y.reshape(-1))
            
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_mt(model, dataloader, device, pad_id_amt, pad_id_ingred, criterion_amt, criterion_ingred):
    model.eval()  # turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for x_amt, x_ingred, y_amt, y_ingred in dataloader:
            x_amt = x_amt.to(device).transpose(0, 1)   # (seq_len, batch)
            x_ingred = x_ingred.to(device).transpose(0, 1)
            y_amt = y_amt.to(device).transpose(0, 1)
            y_ingred = y_ingred.to(device).transpose(0, 1)

            # padding mask
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
            
            total_loss += loss_amt.item() + loss_ingred.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
