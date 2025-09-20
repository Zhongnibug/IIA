import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
    
# Model Architecture
class IncrementalInformationAware(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 generator, iia_encoder,
                 iia_embed, iia_pos):
        super(IncrementalInformationAware, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

        # iia encoder use transformer decoder
        self.iia_encoder = iia_encoder
        self.iia_embed = iia_embed
        self.iia_pos = iia_pos

    def forward(self, src, low_tgt, high_tgt,
                low_src_key_padding_mask=None, 
                high_src_key_padding_mask=None,
                tgt_attn_mask=None, **kwargs):
        "Take in and process masked src and target sequences."
        memory, memory_incre, memory_mix = self.encode_mix(src, key_padding_mask=low_src_key_padding_mask)


        res_low = self.decode(memory, low_tgt,
                              src_key_padding_mask=low_src_key_padding_mask,
                              tgt_attn_mask=tgt_attn_mask)
        res_high = self.decode(memory_mix, high_tgt,
                              src_key_padding_mask=high_src_key_padding_mask,
                              tgt_attn_mask=tgt_attn_mask)
        return res_low, res_high
    
    def iia(self, src, key_padding_mask=None, attn_mask=None, **kwargs):
        src_embedds = self.iia_embed(src)
        src_embedds = self.iia_pos(src_embedds)
        out = self.iia_encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return out
        
    def encode(self, src, key_padding_mask=None, attn_mask=None, **kwargs):
        src_embedds = self.src_embed(src)
        src_embedds = self.src_pos(src_embedds)
        return self.encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    
    def encode_mix(self, src, key_padding_mask=None, attn_mask=None, **kwargs):
        out1 = self.encode(src, key_padding_mask=key_padding_mask)
        out2 = self.iia(src, key_padding_mask=key_padding_mask)
        return out1, out2, torch.cat((out1, out2), dim=1)
    
    def decode(self, memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None, **kwargs):
        target_embedds = self.tgt_embed(tgt)
        target_embedds = self.tgt_pos(target_embedds)
        out = self.decoder(target_embedds, memory,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_attn_mask=tgt_attn_mask)
        return self.generator(out)

class IncrementalInformationAwareFussion(nn.Module):
    def __init__(self, struct_model, seman_model, struct_ratio=0.5):
        super(IncrementalInformationAwareFussion, self).__init__()
        self.struct_model = struct_model
        self.seman_model = seman_model
        self.struct_ratio = struct_ratio
        
    def forward(self, src, low_tgt, high_tgt,
                low_src_key_padding_mask=None, 
                high_src_key_padding_mask=None,
                tgt_attn_mask=None, **kwargs):
        "Take in and process masked src and target sequences."
        struct_memory, struct_memory_incre, struct_memory_mix, \
            seman_memory, seman_memory_incre, seman_memory_mix = self.encode_mix(src, key_padding_mask=low_src_key_padding_mask)
        res_struct_low = self.struct_model.decode(struct_memory, low_tgt,
                                                  src_key_padding_mask=low_src_key_padding_mask,
                                                  tgt_attn_mask=tgt_attn_mask)
        res_struct_high = self.struct_model.decode(struct_memory_mix, high_tgt,
                                                   src_key_padding_mask=high_src_key_padding_mask,
                                                   tgt_attn_mask=tgt_attn_mask)
        res_seman_low = self.seman_model.decode(seman_memory, low_tgt,
                                                src_key_padding_mask=low_src_key_padding_mask,
                                                tgt_attn_mask=tgt_attn_mask)
        res_seman_high = self.seman_model.decode(seman_memory_mix, high_tgt,
                                                 src_key_padding_mask=high_src_key_padding_mask,
                                                 tgt_attn_mask=tgt_attn_mask)
        return res_struct_low, res_struct_high, res_seman_low, res_seman_high
    
    def encode(self, src, key_padding_mask=None, attn_mask=None, **kwargs):
        struct_memory, struct_memory_incre, struct_memory_mix = self.struct_model.encode_mix(src, key_padding_mask=key_padding_mask)
        seman_memory, seman_memory_incre, seman_memory_mix = self.seman_model.encode_mix(src, key_padding_mask=key_padding_mask)
        return struct_memory, struct_memory_incre, struct_memory_mix, \
                seman_memory, seman_memory_incre, seman_memory_mix
    
    def decode(self, struct_memory, seman_memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None, **kwargs):
        res_struct = self.struct_model.decode(struct_memory, tgt,
                                              src_key_padding_mask=src_key_padding_mask,
                                              tgt_attn_mask=tgt_attn_mask)
        res_seman = self.seman_model.decode(seman_memory, tgt,
                                            src_key_padding_mask=src_key_padding_mask,
                                            tgt_attn_mask=tgt_attn_mask)

        return self.struct_ratio * res_struct + (1 - self.struct_ratio) * res_seman

if __name__ == '__main__':
    pass