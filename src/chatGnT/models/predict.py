import torch


def predict(model, device, pad_id, vocab, inv_vocab, start_ingred, max_len=50, temperature=0.8):
    model.eval()

    # Start sequence with group of tokens: amt, unit, ingred, sep
    ids = [vocab[i] for i in start_ingred]

    for _ in range(max_len):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)

        # padding mask
        pad_mask = (x == pad_id).transpose(0, 1)  # (batch, seq_len)

        # causal mask
        seq_len = x.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        with torch.no_grad():
            output = model(
                src=x,
                src_key_padding_mask=pad_mask,
                src_mask=src_mask
            )

            logits = output[-1, 0, :]

            # block repeated <sep>
            last_token = inv_vocab[ids[-1]]
            if last_token == "<sep>":
                logits[vocab["<sep>"]] = -float("inf")

            # Sampling for diversity
            # Convert to probabilities with temperature
            probs = torch.softmax(logits / temperature, dim=-1)
            # Sample for next token
            next_id = torch.multinomial(probs, num_samples=1).item()

        # Append prediction
        ids.append(next_id)

        # Stop if end token
        if inv_vocab[next_id] == "<end>":
            break

    # Convert back to tokens
    tokens = [inv_vocab[i] for i in ids]
    return tokens


def predict_2head(model, device, pad_id_amt, pad_id_ingred, vocab_amt, vocab_ingred, inv_vocab_amt, inv_vocab_ingred, start_tokens, max_len=50, temperature=0.8):
    model.eval()

    # Start sequence with group of tokens: amt, unit, ingred, sep
    amt_ids = [vocab_amt[amt] for amt, _ in start_tokens]
    ingred_ids = [vocab_ingred[ingred] for _, ingred in start_tokens]

    for _ in range(max_len):
        x_amt = torch.tensor(amt_ids, dtype=torch.long, device=device).unsqueeze(1)     # (seq_len, 1)
        x_ingred = torch.tensor(ingred_ids, dtype=torch.long, device=device).unsqueeze(1)

        # padding mask
        pad_mask = (
            (x_amt == pad_id_amt) |
            (x_ingred == pad_id_ingred)
        ).transpose(0, 1)
        
        # causal mask
        seq_len = x_amt.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        with torch.no_grad():
            output_amt, output_ingred = model(
                src_amt=x_amt,
                src_ingred=x_ingred,
                src_key_padding_mask=pad_mask,
                src_mask=src_mask
            )

            logits_amt = output_amt[-1, 0, :]
            logits_ingred = output_ingred[-1, 0, :]

            # Sampling for diversity
            # Convert to probabilities with temperature
            probs_amt = torch.softmax(logits_amt / temperature, dim=-1)
            probs_ingred = torch.softmax(logits_ingred / temperature, dim=-1)
            # Sample for next token
            next_id_amt = torch.multinomial(probs_amt, num_samples=1).item()
            next_id_ingred = torch.multinomial(probs_ingred, num_samples=1).item()

        # Append prediction
        amt_ids.append(next_id_amt)
        ingred_ids.append(next_id_ingred)

        # Stop if end token
        if inv_vocab_amt[next_id_amt] == "<end>" or inv_vocab_ingred[next_id_ingred] == "<end>":
            break

    # Convert back to tokens
    tokens = [(inv_vocab_amt[i], inv_vocab_ingred[j]) for i, j in zip(amt_ids, ingred_ids)]
    return tokens

