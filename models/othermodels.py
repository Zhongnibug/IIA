import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from omegaconf import OmegaConf

class SimpleChangeShape(nn.Module):
    def __init__(self, original_shape, changed_shape):
        super(SimpleChangeShape, self).__init__()
        self.in_dim = torch.prod(torch.tensor(original_shape))
        self.out_dim = changed_shape
        self.original_shape_len = len(original_shape)
        self.fc = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        batchshape = list(x.shape[:-self.original_shape_len])
        x = x.view(-1, self.in_dim)
        x = self.fc(x)
        x = x.view(*batchshape, self.out_dim)

        return x
    

# No sqrt d_moel mutiply
class EmbeddingsNOMUTI(nn.Module):
    def __init__(self, d_model, vocab):
        super(EmbeddingsNOMUTI, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

    def forward(self, x):
        embedds = self.lut(x)
        return embedds
    
class RandomNoisePosition(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_pos_len=1000, alpha=0.1, mean=0.0, std_dev=1.0):
        super(RandomNoisePosition, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_pos_len, d_model, requires_grad=False)
        position = torch.arange(0, max_pos_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        alpha = torch.tensor(alpha)
        self.register_buffer('alpha', alpha)
        mean = torch.tensor(mean)
        self.register_buffer('mean', mean)
        std_dev = torch.tensor(std_dev)
        self.register_buffer('std_dev', std_dev)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)

        x = x + (torch.randn_like(x) * self.std_dev + self.mean)*self.alpha
        return x

class RandomNoiseHighFrequenceEmbeddings(nn.Module):
    def __init__(self, d_model, vocab, alpha=0.1, mean=0.0, std_dev=1.0):
        super(RandomNoiseHighFrequenceEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

        alpha = torch.tensor(alpha)
        self.register_buffer('alpha', alpha)
        mean = torch.tensor(mean)
        self.register_buffer('mean', mean)
        std_dev = torch.tensor(std_dev)
        self.register_buffer('std_dev', std_dev)

    def forward(self, x, noise_mask=None):
        embedds = self.lut(x)
        embedds = embedds * math.sqrt(self.d_model)
        
        if noise_mask is not None:
            noise = (torch.randn_like(embedds) * self.std_dev + self.mean)*self.alpha
            masked_noise = noise * (~noise_mask).unsqueeze(-1)  
            return (embedds+ masked_noise) * math.sqrt(self.d_model)
        else:
            return embedds * math.sqrt(self.d_model)

# No sqrt d_moel mutiply
class RandomNoiseHighFrequenceEmbeddingsNOMUTI(nn.Module):
    def __init__(self, d_model, vocab, alpha=0.1, mean=0.0, std_dev=1.0):
        super(RandomNoiseHighFrequenceEmbeddingsNOMUTI, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

        alpha = torch.tensor(alpha)
        self.register_buffer('alpha', alpha)
        mean = torch.tensor(mean)
        self.register_buffer('mean', mean)
        std_dev = torch.tensor(std_dev)
        self.register_buffer('std_dev', std_dev)

    def forward(self, x, noise_mask=None):
        embedds = self.lut(x)
        embedds = embedds * math.sqrt(self.d_model)
        
        if noise_mask is not None:
            noise = (torch.randn_like(embedds) * self.std_dev + self.mean)*self.alpha
            masked_noise = noise * (~noise_mask).unsqueeze(-1)  
            return (embedds+ masked_noise) * math.sqrt(self.d_model)
        else:
            return embedds * math.sqrt(self.d_model)   

# Random Noise High Frequence Model Architecture
class RandomNoiseHighFrequenceTransformer(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 generator):
        super(RandomNoiseHighFrequenceTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_attn_mask=None, src_noise_mask=None, tgt_noise_mask=None):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, key_padding_mask=src_key_padding_mask, src_noise_mask=src_noise_mask)
        res = self.decode(memory, tgt,
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_attn_mask=tgt_attn_mask,
                          tgt_noise_mask=tgt_noise_mask)
        return res
    
    def encode(self, src, key_padding_mask=None, attn_mask=None, src_noise_mask=None):
        if src_noise_mask is None:
            src_embedds = self.src_embed(src)
        else:
            src_embedds = self.src_embed(src, src_noise_mask)
        return self.encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    
    def decode(self, memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None, tgt_noise_mask=None):
        if tgt_noise_mask is None:
            target_embedds = self.tgt_embed(tgt)
        else:
            target_embedds = self.tgt_embed(tgt, tgt_noise_mask)
        out = self.decoder(target_embedds, memory,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_attn_mask=tgt_attn_mask)
        return self.generator(out)
    

# Add only global low frequence words as noise
class GlobalLowFreqNoiseByHighFreqEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(GlobalLowFreqNoiseByHighFreqEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

    def forward(self, x, reverse_high_freqs=None, low_freqs=None, high_freqs=None):
        r'''
        Args:
            reverse_high_freqs(tensor):  i == high_freqs[vid][reverse_high_freqs[i]] if reverse_high_freqs[vid][i] != -1.
            low_freqs(tensor): Low-frequence word indexes of vocaburary of the whole batch.
            high_freqs(tensor): High-frequence word indexes of vocaburary of the whole the batch.  
        '''
        embedds = self.lut(x)

        if reverse_high_freqs is not None:
            low_freq_embedds = self.lut(low_freqs)
            high_freq_embedds = self.lut(high_freqs)
            dot_products = torch.mm(low_freq_embedds, high_freq_embedds.t())

            norm_low_freq = torch.norm(low_freq_embedds, dim=1, keepdim=True)
            norm_high_freq = torch.norm(high_freq_embedds, dim=1, keepdim=True)
            norm_products = torch.mm(norm_low_freq, norm_high_freq.t())

            cosine_similarities = dot_products / norm_products

            max_values, max_indices = torch.max(cosine_similarities, dim=1)

            result = torch.full([cosine_similarities.shape[0]+1,cosine_similarities.shape[1]], float('-inf'))
            result[torch.arange(result.size(0)-1), max_indices] = cosine_similarities[torch.arange(cosine_similarities.size(0)), max_indices]
            result[-1, :] = -10000.0

            result = result.t()
            softmax_res = torch.nn.functional.softmax(result, dim=1)
            softmax_res_del_tail = softmax_res[:,:-1]

            noise_embedds = torch.zeros([softmax_res_del_tail.shape[0]+1, self.d_model], dtype=torch.float)
            noise_embedds[:-1, :] = torch.mm(softmax_res_del_tail, low_freq_embedds)

            embedds = self.lut(x) + noise_embedds[reverse_high_freqs[x]]

            return embedds
        else:
            return embedds
        
# Global High Frequence add Global Low frequence noise Model Architecture
class GlobalLowFreqNoiseByHighFreqTransformer(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 generator):
        super(GlobalLowFreqNoiseByHighFreqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_attn_mask=None, 
                src_reverse_high_freqs=None, src_low_freqs=None, src_high_freqs=None,
                tgt_reverse_high_freqs=None, tgt_low_freqs=None, tgt_high_freqs=None):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, key_padding_mask=src_key_padding_mask,
                             src_reverse_high_freqs=src_reverse_high_freqs,
                             src_low_freqs=src_low_freqs,
                             src_high_freqs=src_high_freqs)
        res = self.decode(memory, tgt,
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_attn_mask=tgt_attn_mask,
                          tgt_reverse_high_freqs=tgt_reverse_high_freqs,
                          tgt_low_freqs=tgt_low_freqs,
                          tgt_high_freqs=tgt_high_freqs)
        return res
    
    def encode(self, src, key_padding_mask=None, attn_mask=None,
               src_reverse_high_freqs=None, src_low_freqs=None, src_high_freqs=None):
        if src_reverse_high_freqs is None:
            src_embedds = self.src_embed(src)
        else:
            src_embedds = self.src_embed(src,
                                         src_reverse_high_freqs,
                                         src_low_freqs,
                                         src_high_freqs)
        return self.encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    
    def decode(self, memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None,
               tgt_reverse_high_freqs=None, tgt_low_freqs=None, tgt_high_freqs=None):
        if tgt_reverse_high_freqs is None:
            target_embedds = self.tgt_embed(tgt)
        else:
            target_embedds = self.tgt_embed(tgt,
                                            tgt_reverse_high_freqs,
                                            tgt_low_freqs,
                                            tgt_high_freqs)
        out = self.decoder(target_embedds, memory,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_attn_mask=tgt_attn_mask)
        return self.generator(out)
    

# Add low frequence words as noise
class LowFreqNoiseByHighFreqEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(LowFreqNoiseByHighFreqEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

    def forward(self, x, reverse_high_freqs=None, 
                common_low_freqs=None, common_high_freqs=None,
                low_freqs_masks=None, high_freqs_masks=None):
        r'''
        Args:
            reverse_high_freqs(tensor):  i == common_high_freqs[reverse_high_freqs[s][i]] 
                                  if reverse_high_freqs[s][i] != -1. Shape == [bs, vocab].
            common_low_freqs(tensor): Low-frequence word indexes of the batch. Shape == [low_freq].
            common_high_freqs(tensor): High-frequence word indexes of the batch. Shape == [high_freq].
            low_freqs_masks(tensor): Low-frequence word mask to common_low_freqs. Shape == [bs, low_freq].
            high_freqs_masks(tensor): High-frequence word mask to common_low_freqs. Shape == [bs, high_freq].
        '''

        if reverse_high_freqs is not None:

            low_freq_embedds = self.lut(common_low_freqs) # (low_freq, d_model)
            high_freq_embedds = self.lut(common_high_freqs)  # (high_freq, d_model)
            dot_products = torch.mm(high_freq_embedds, low_freq_embedds.t()) # (high_freq, low_freq)

            norm_low_freq = torch.norm(low_freq_embedds, dim=1, keepdim=True) # (low_freq, 1)

            norm_high_freq = torch.norm(high_freq_embedds, dim=1, keepdim=True) # (high_freq, 1)

            norm_products = torch.mm(norm_high_freq, norm_low_freq.t()) # (high_freq, low_freq)

            cosine_similarities = dot_products / norm_products # (high_freq, low_freq)


            
            masked_cosine_similarities = torch.where(high_freqs_masks.unsqueeze(-1), torch.tensor(-float('inf')), cosine_similarities) # (bs,high_freq,low_freq)

            masked_cosine_similarities.transpose_(-2,-1) # (bs, low_freq, high_freq)
            masked_cosine_similarities = torch.where(low_freqs_masks.unsqueeze(-1), torch.tensor(-float('inf')), masked_cosine_similarities)  # (bs, low_freq, high_freq)

            masked_sim_shape = masked_cosine_similarities.shape # (bs, low_freq, high_freq)
            max_indices = torch.argmax(masked_cosine_similarities, dim=-1) # (bs, low_freq)

            masked_cosine_similarities = masked_cosine_similarities.reshape(-1, masked_sim_shape[2]) # (bs*low_freq, high_freq)
            max_indices = max_indices.view(-1) # (bs*low_freq)

            max_mask = torch.zeros_like(masked_cosine_similarities, dtype=torch.bool) # (bs*low_freq, high_freq)
            max_mask[torch.arange(masked_cosine_similarities.shape[0]), max_indices] = True
            masked_cosine_similarities[~max_mask] = float('-inf')
            masked_cosine_similarities = masked_cosine_similarities.view(masked_sim_shape[0], masked_sim_shape[1], masked_sim_shape[2]) # (bs, low_freq, high_freq)
            
            masked_cosine_similarities.transpose_(1, 2)  # (bs, high_freq, low_freq)
            masked_cosine_similarities = torch.cat((masked_cosine_similarities, torch.full([masked_sim_shape[0], masked_sim_shape[2], 1], -10000.).to(masked_cosine_similarities.device)), dim=-1) # (bs, high_freq, low_freq+1)

            masked_cosine_similarities = torch.nn.functional.softmax(masked_cosine_similarities, dim=-1) # (bs, high_freq, low_freq +1)
            masked_cosine_similarities = masked_cosine_similarities[:,:,:-1] # (bs, high_freq, low_freq)

            noise_embedds = torch.matmul(masked_cosine_similarities, low_freq_embedds) # (bs, high_freq, d_model)
            noise_embedds = torch.cat((noise_embedds, torch.zeros([masked_sim_shape[0], 1, self.d_model]).to(noise_embedds.device)), dim=-2)

            interleave_index = torch.arange(masked_sim_shape[0]).repeat_interleave(x.shape[1]) # (bs*batch_max_len)

            embedds = self.lut(x) + noise_embedds[interleave_index, reverse_high_freqs[interleave_index, x.view(-1)]].view(masked_sim_shape[0], -1, self.d_model)
            # (bs, batch_max_len, d_model)

            return embedds * math.sqrt(self.d_model)
        else: 
            embedds = self.lut(x)
            return embedds * math.sqrt(self.d_model)
        
# Add low frequence words as noise and No sqrt d_moel mutiply
class LowFreqNoiseByHighFreqEmbeddingsNOMUTI(nn.Module):
    def __init__(self, d_model, vocab):
        super(LowFreqNoiseByHighFreqEmbeddingsNOMUTI, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model =d_model

    def forward(self, x, reverse_high_freqs=None, 
                common_low_freqs=None, common_high_freqs=None,
                low_freqs_masks=None, high_freqs_masks=None):
        r'''
        Args:
            reverse_high_freqs(tensor):  i == common_high_freqs[reverse_high_freqs[s][i]] 
                                  if reverse_high_freqs[s][i] != -1. Shape == [bs, vocab].
            common_low_freqs(tensor): Low-frequence word indexes of the batch. Shape == [low_freq].
            common_high_freqs(tensor): High-frequence word indexes of the batch. Shape == [high_freq].
            low_freqs_masks(tensor): Low-frequence word mask to common_low_freqs. Shape == [bs, low_freq].
            high_freqs_masks(tensor): High-frequence word mask to common_low_freqs. Shape == [bs, high_freq].
        '''

        if reverse_high_freqs is not None:

            low_freq_embedds = self.lut(common_low_freqs) # (low_freq, d_model)
            high_freq_embedds = self.lut(common_high_freqs)  # (high_freq, d_model)
            dot_products = torch.mm(high_freq_embedds, low_freq_embedds.t()) # (high_freq, low_freq)

            norm_low_freq = torch.norm(low_freq_embedds, dim=1, keepdim=True) # (low_freq, 1)

            norm_high_freq = torch.norm(high_freq_embedds, dim=1, keepdim=True) # (high_freq, 1)

            norm_products = torch.mm(norm_high_freq, norm_low_freq.t()) # (high_freq, low_freq)

            cosine_similarities = dot_products / norm_products # (high_freq, low_freq)


            
            masked_cosine_similarities = torch.where(high_freqs_masks.unsqueeze(-1), torch.tensor(-float('inf')), cosine_similarities) # (bs, high_freq, low_freq)

            masked_cosine_similarities.transpose_(-2,-1) # (bs, low_freq, high_freq)
            masked_cosine_similarities = torch.where(low_freqs_masks.unsqueeze(-1), torch.tensor(-float('inf')), masked_cosine_similarities)  # (bs, low_freq, high_freq)

            masked_sim_shape = masked_cosine_similarities.shape # (bs, low_freq, high_freq)
            max_indices = torch.argmax(masked_cosine_similarities, dim=-1) # (bs, low_freq)

            masked_cosine_similarities = masked_cosine_similarities.reshape(-1, masked_sim_shape[2]) # (bs*low_freq, high_freq)
            max_indices = max_indices.view(-1) # (bs*low_freq)

            max_mask = torch.zeros_like(masked_cosine_similarities, dtype=torch.bool) # (bs*low_freq, high_freq)
            max_mask[torch.arange(masked_cosine_similarities.shape[0]), max_indices] = True
            masked_cosine_similarities[~max_mask] = float('-inf')
            masked_cosine_similarities = masked_cosine_similarities.view(masked_sim_shape[0], masked_sim_shape[1], masked_sim_shape[2]) # (bs, low_freq, high_freq)
            
            masked_cosine_similarities.transpose_(1, 2)  # (bs, high_freq, low_freq)
            masked_cosine_similarities = torch.cat((masked_cosine_similarities, torch.full([masked_sim_shape[0], masked_sim_shape[2], 1], -10000.).to(masked_cosine_similarities.device)), dim=-1) # (bs, high_freq, low_freq+1)

            masked_cosine_similarities = torch.nn.functional.softmax(masked_cosine_similarities, dim=-1) # (bs, high_freq, low_freq +1)
            masked_cosine_similarities = masked_cosine_similarities[:,:,:-1] # (bs, high_freq, low_freq)

            noise_embedds = torch.matmul(masked_cosine_similarities, low_freq_embedds) # (bs, high_freq, d_model)
            noise_embedds = torch.cat((noise_embedds, torch.zeros([masked_sim_shape[0], 1, self.d_model]).to(noise_embedds.device)), dim=-2)

            interleave_index = torch.arange(masked_sim_shape[0]).repeat_interleave(x.shape[1]) # (bs*batch_max_len)

            embedds = self.lut(x) + noise_embedds[interleave_index, reverse_high_freqs[interleave_index, x.view(-1)]].view(masked_sim_shape[0], -1, self.d_model)
            # (bs, batch_max_len, d_model)

            return embedds
        else: 
            embedds = self.lut(x)
            return embedds

# High Frequence add Low frequence noise Model Architecture
class LowFreqNoiseByHighFreqTransformer(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 generator):
        super(LowFreqNoiseByHighFreqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_attn_mask=None, 
                src_reverse_high_freqs=None, src_common_low_freqs=None, src_common_high_freqs=None,
                src_low_freqs_masks=None, src_high_freqs_masks=None,
                tgt_reverse_high_freqs=None, tgt_common_low_freqs=None, tgt_common_high_freqs=None,
                tgt_low_freqs_masks=None, tgt_high_freqs_masks=None):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, key_padding_mask=src_key_padding_mask,
                             src_reverse_high_freqs=src_reverse_high_freqs,
                             src_common_low_freqs=src_common_low_freqs,
                             src_common_high_freqs=src_common_high_freqs,
                             src_low_freqs_masks=src_low_freqs_masks,
                             src_high_freqs_masks=src_high_freqs_masks)
        res = self.decode(memory, tgt,
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_attn_mask=tgt_attn_mask,
                          tgt_reverse_high_freqs=tgt_reverse_high_freqs,
                          tgt_common_low_freqs=tgt_common_low_freqs,
                          tgt_common_high_freqs=tgt_common_high_freqs,
                          tgt_low_freqs_masks=tgt_low_freqs_masks,
                          tgt_high_freqs_masks=tgt_high_freqs_masks)
        return res
    
    def encode(self, src, key_padding_mask=None, attn_mask=None,
               src_reverse_high_freqs=None, src_common_low_freqs=None, src_common_high_freqs=None,
               src_low_freqs_masks=None, src_high_freqs_masks=None):
        if src_reverse_high_freqs is None:
            src_embedds = self.src_embed(src)
        else:
            src_embedds = self.src_embed(src,
                                         src_reverse_high_freqs,
                                         src_common_low_freqs,
                                         src_common_high_freqs,
                                         src_low_freqs_masks,
                                         src_high_freqs_masks)
        return self.encoder(src_embedds, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    
    def decode(self, memory, tgt, src_key_padding_mask=None, tgt_attn_mask=None,
               tgt_reverse_high_freqs=None, tgt_common_low_freqs=None, tgt_common_high_freqs=None,
               tgt_low_freqs_masks=None, tgt_high_freqs_masks=None):
        if tgt_reverse_high_freqs is None:
            target_embedds = self.tgt_embed(tgt)
        else:
            target_embedds = self.tgt_embed(tgt,
                                            tgt_reverse_high_freqs,
                                            tgt_common_low_freqs,
                                            tgt_common_high_freqs,
                                            tgt_low_freqs_masks,
                                            tgt_high_freqs_masks)
        out = self.decoder(target_embedds, memory,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_attn_mask=tgt_attn_mask)
        return self.generator(out)

if __name__ == '__main__':
    pass