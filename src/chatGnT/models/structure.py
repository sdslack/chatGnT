import torch

def _build_single_task_allowed_masks(vocab, device):
    vocab_size = len(vocab)
    allow_amt = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    allow_ingred = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    for token, token_id in vocab.items():
        if not isinstance(token, str):
            continue
        if token.startswith("<amt>"):
            allow_amt[token_id] = True
        elif token.startswith("<ingred>"):
            allow_ingred[token_id] = True

    allow_amt_or_end = allow_amt.clone()
    allow_amt_or_end[vocab["<end>"]] = True
    return allow_ingred, allow_amt_or_end


def mask_single_task_output_logits(output, vocab):
    """
    Restrict single-task logits so the model alternates:
    ingred, amt, ingred, amt/end, ...

    `output` is shaped `(seq_len, batch, vocab_size)` and predicts the next
    token at each timestep.
    """
    allow_ingred, allow_amt_or_end = _build_single_task_allowed_masks(
        vocab, output.device
    )
    seq_len = output.size(0)
    expect_ingred = (torch.arange(seq_len, device=output.device) % 2 == 0)
    allowed = torch.where(
        expect_ingred.unsqueeze(-1),
        allow_ingred.unsqueeze(0),
        allow_amt_or_end.unsqueeze(0),
    )
    return output.masked_fill(~allowed.unsqueeze(1), float("-inf"))


def mask_single_task_next_logits(logits, vocab, current_length):
    """
    Restrict next-token logits for single-task generation based on sequence
    parity. `current_length` is the number of tokens already generated.
    """
    allow_ingred, allow_amt_or_end = _build_single_task_allowed_masks(
        vocab, logits.device
    )
    allowed = allow_ingred if current_length % 2 == 1 else allow_amt_or_end
    return logits.masked_fill(~allowed, float("-inf"))
