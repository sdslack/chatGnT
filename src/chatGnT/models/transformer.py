import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding

# Based on tutorial https://h-huang.github.io/tutorials/beginner/transformer_tutorial.html



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    # Mask to block attention to future tokens
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(output)
        return output


class TransformerModel2Head(nn.Module):

    def __init__(self, ntoken_amt, ntoken_ingred, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel2Head, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Shared embedding for amt and ingred since tokens encode information
        self.encoder = nn.Embedding(ntoken_amt + ntoken_ingred, ninp)
        self.ninp = ninp

        # Separate output heads
        self.amt_head = nn.Linear(ninp, ntoken_amt)
        # self.amt_head = nn.Linear(ninp + ntoken_ingred, ntoken_amt)  # amt depends on ingred
        self.ingred_head = nn.Linear(ninp, ntoken_ingred)

        self.init_weights()

    # Mask to block attention to future tokens
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.amt_head.weight.data.uniform_(-initrange, initrange)
        self.amt_head.bias.data.zero_()

        self.ingred_head.weight.data.uniform_(-initrange, initrange)
        self.ingred_head.bias.data.zero_()

    def forward(self, src_amt, src_ingred, src_key_padding_mask, src_mask):
        # Embed both amt and ingred
        amt_emb = self.encoder(src_amt) * math.sqrt(self.ninp)
        ingred_emb = self.encoder(src_ingred) * math.sqrt(self.ninp)

        # Combined amt and ingred to joint token representation
        src = amt_emb + ingred_emb

        # Positional encoding
        src = self.pos_encoder(src)

        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Predict both amt and ingred at every step,
        output_ingred = self.ingred_head(output)
        output_amt = self.amt_head(output)

        # Previous version conditioning amt on the full distribution over ingredients
        # probs_ingred = torch.softmax(output_ingred, dim=-1)
        # amt_input = torch.cat([output, probs_ingred], dim=-1)
        # output_amt = self.amt_head(amt_input)

        return output_amt, output_ingred
