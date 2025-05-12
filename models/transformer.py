import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

# Model Architecture
class Transformer(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_attn_mask=None, **kwargs):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, key_padding_mask=src_key_padding_mask)
        res = self.decode(memory, tgt,
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_attn_mask=tgt_attn_mask)
        return res
    
    def encode(self, src, key_padding_mask=None, attn_mask=None, **kwargs):
        src_embedds = self.src_embed(src)
        src_embedds = self.src_pos(src_embedds)
        return self.encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    
    def decode(self, memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None, **kwargs):
        target_embedds = self.tgt_embed(tgt)
        target_embedds = self.tgt_pos(target_embedds)
        out = self.decoder(target_embedds, memory,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_attn_mask=tgt_attn_mask)
        return self.generator(out)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

    def forward(self, x):
        embedds = self.lut(x)
        return embedds * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_pos_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_pos_len, d_model, requires_grad=False)
        position = torch.arange(0, max_pos_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        return self.attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
    
class Encoder(nn.Module):
    """
    Encoder
    The encoder is composed of a stack of N=6 identical layers.
    """
    def __init__(self, layer, num_encoder_layers,
                 normalize_before=False,
                 norm=None):
        super(Encoder, self).__init__()
        self.layers = _clones(layer, num_encoder_layers)
        self.normalize_before = normalize_before
        if not normalize_before:
            self.norm = None
        elif norm is None:
            self.norm = nn.LayerNorm(layer.d_model)
        else:
            self.norm = norm
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        return self.norm(x) if self.normalize_before else x

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, num_decoder_layers,
                 normalize_before=False,
                 norm=None):
        super(Decoder, self).__init__()
        self.layers = _clones(layer, num_decoder_layers)
        self.normalize_before = normalize_before
        if not normalize_before:
            self.norm = None
        elif norm is None:
            self.norm = nn.LayerNorm(layer.d_model)
        else:
            self.norm = norm
        
    def forward(self, x, memory, src_key_padding_mask=None, tgt_attn_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_key_padding_mask=src_key_padding_mask, tgt_attn_mask=tgt_attn_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.norm(x) if self.normalize_before else x
    
class EncoderLayer(nn.Module):
    "EncoderLayer is made up of two sublayer: self-attn and feed forward"                                                                                                         
    def __init__(self, d_model, self_attn, feed_forward, connection):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.connection = _clones(connection, 2)
        self.d_model = d_model

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # attention sub layer
        x = self.connection[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask))
        # feed forward sub layer
        x = self.connection[1](x, self.feed_forward)
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, self_attn, src_attn, feed_forward, connection):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.connection = _clones(connection, 3)
 
    def forward(self, x, memory, src_key_padding_mask=None, tgt_attn_mask=None, tgt_key_padding_mask=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.connection[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_attn_mask))
        x = self.connection[1](x, lambda x: self.src_attn(x, m, m, key_padding_mask=src_key_padding_mask))
        return self.connection[2](x, self.feed_forward)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _activation_fn(activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):

    def __init__(self, d_model, dropout,
                 normalize_before=False,
                 norm=None):
        super(SublayerConnection, self).__init__()
        if norm is None:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    # original transformer
    def post_forward(self, x, sublayer):
        sublayer_out = sublayer(x)
        x_norm = self.norm(x + self.dropout(sublayer_out))
        return x_norm

    # take norm to the front of sublayer
    def pre_forward(self, x, sublayer):
        norm_out = self.norm(x)
        sublayer_out = sublayer(norm_out)
        x_norm = self.norm(x + self.dropout(sublayer_out))
        return x_norm

    def forward(self, x, sublayer):
        return self.pre_forward(x, sublayer) if self.normalize_before else self.post_forward(x, sublayer)

def _clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    pass