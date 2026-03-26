import torch

def predict(model, device, pad_id, vocab,inv_vocab, ids, max_len=50):
    model.eval()

    for _ in range(max_len):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)

        seq_len = x.size(0)
        # pad_mask = (x == pad_id).transpose(0, 1)
        # src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
        #TODO: do I want these for prediction?

        with torch.no_grad():
            output = model(
                src=x,
                src_key_padding_mask=None,
                src_mask=None
            )

            logits = output[-1, 0, :]
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Get most likely next token
            #TODO: how to introduce temperature here?
            next_id = torch.argmax(probs).item()

        # Append prediction
        ids.append(next_id)

        # Stop if end token
        if inv_vocab[next_id] == "<end>":
            break

    # Convert back to tokens
    tokens = [inv_vocab[i] for i in ids]
    return tokens
