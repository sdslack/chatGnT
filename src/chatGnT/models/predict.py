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



def predict_groups(model, device, pad_id, vocab, inv_vocab, start_ingred, max_groups=10, temperature=0.8):
    """
    Generate a recipe conditioned on a starting ingredient.

    Args:
        model: trained TransformerModel
        device: torch device
        pad_id: ID of <pad> token
        vocab: token -> id
        itos: id -> token
        start_ingred: string like '<ingred>lime</ingred>'
        max_groups: maximum number of ingredient groups to generate
        temperature: sampling temperature for diversity

    Returns:
        List of tokens representing the full recipe
    """
    model.eval()
    
    # Start sequence with group of tokens: amt, unit, ingred, sep
    generated_ids = [vocab[i] for i in start_ingred]
    tokens = start_ingred
    
    for _ in range(max_groups):
        group_ids = []
        while True:
            # Current input sequence
            x = torch.tensor(generated_ids + group_ids, dtype=torch.long, device=device).unsqueeze(1)
            seq_len = x.size(0)
            
            # padding mask
            pad_mask = (x == pad_id).transpose(0, 1)  # (batch, seq_len)

            # causal mask
            seq_len = x.size(0)
            src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            
            with torch.no_grad():
                output = model(
                    src=x,
                    src_key_padding_mask=pad_mask,
                    src_mask=src_mask)

                logits = output[-1, 0, :]
                
                # Sampling for diversity
                # Convert to probabilities with temperature
                probs = torch.softmax(logits / temperature, dim=-1)
                # Sample for next token
                next_id = torch.multinomial(probs, num_samples=1).item()
            
            token = inv_vocab[next_id]
            group_ids.append(next_id)
            
            # Stop generating this group when we reach <sep> or <end>
            if token in ["<sep>", "<end>"]:
                break
        
        # Append the completed group to the sequence
        generated_ids.extend(group_ids)
        tokens.extend([inv_vocab[i] for i in group_ids])
        
        if token == "<end>":
            break
    
    return tokens