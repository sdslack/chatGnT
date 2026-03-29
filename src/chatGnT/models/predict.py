import torch

def predict(model, device, pad_id, vocab, inv_vocab, ids, max_len=50):
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
    
    # Start sequence with the ingredient
    generated_ids = [vocab[start_ingred]]
    tokens = [start_ingred]
    
    for _ in range(max_groups):
        group_ids = []
        while True:
            # Current input sequence
            x = torch.tensor(generated_ids + group_ids, dtype=torch.long, device=device).unsqueeze(1)
            seq_len = x.size(0)
            
            # pad_mask = (x == pad_id).transpose(0, 1)
            # src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            
            with torch.no_grad():
                output = model(
                    src=x,
                    src_key_padding_mask=None,
                    src_mask=None)

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
    
    # Optional: clean repeated <sep>
    clean_tokens = []
    for t in tokens:
        if clean_tokens and t == "<sep>" and clean_tokens[-1] == "<sep>":
            continue
        clean_tokens.append(t)
    
    return clean_tokens