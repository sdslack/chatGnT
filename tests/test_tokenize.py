from chatGnT.data.tokenize import find_long_decimal_tokens


def test_find_long_decimal_tokens_flags_unrounded_values():
    vocab = {
        "<amt>0.16666666666666666 glass</amt>": 1,
        "<amt>0.3333333333333333 part</amt>": 2,
        "<amt>0.167 glass</amt>": 3,
        "<ingred>gin</ingred>": 4,
        "<pad>": 0,
    }

    flagged = find_long_decimal_tokens(vocab)

    assert flagged == [
        "<amt>0.16666666666666666 glass</amt>",
        "<amt>0.3333333333333333 part</amt>",
    ]
