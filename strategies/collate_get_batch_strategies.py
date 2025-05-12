import torch
import numpy as np

def subsequent_mask_2d(configs, size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 1

def easy_list(configs, data):
    return list(data)

def easy_stack(configs, data):
    ret_data = torch.stack(data, dim=0)
    return ret_data

def captions_with_pair(configs, batch_size, low_numberic_caps, low_numberic_labels, low_valid_lens,
                       high_numberic_caps, high_numberic_labels, high_valid_lens):
    low_valid_lens = list(low_valid_lens)
    low_valid_lens_sum = sum(low_valid_lens)
    ret_low_caps = torch.stack(low_numberic_caps, dim=0)
    ret_low_labels = torch.stack(low_numberic_labels, dim=0)
    high_valid_lens = list(high_valid_lens)
    high_valid_lens_sum = sum(high_valid_lens)
    ret_high_caps = torch.stack(high_numberic_caps, dim=0)
    ret_high_labels = torch.stack(high_numberic_labels, dim=0)
    captions_masks = subsequent_mask_2d(configs, ret_low_caps.shape[1])

    return ret_low_caps, ret_low_labels, low_valid_lens, low_valid_lens_sum, \
            ret_high_caps, ret_high_labels, high_valid_lens, high_valid_lens_sum, \
            captions_masks

def transformer_sentence_2d_mask(configs, batch_size, numberic_caps, numberic_labels, valid_lens):
    valid_lens = list(valid_lens)
    valid_lens_sum = sum(valid_lens)
    ret_captions = torch.stack(numberic_caps, dim=0)
    ret_labels = torch.stack(numberic_labels, dim=0)
    captions_masks = subsequent_mask_2d(configs, ret_captions.shape[1])
    
    return ret_captions, ret_labels, captions_masks, valid_lens, valid_lens_sum

def features_mask(configs, features, features_masks):
    ret_features =  torch.stack(features, dim=0)
    features_masks = torch.stack(features_masks, dim=0)
    
    return ret_features, features_masks

def masks_extend(configs, masks, extend_len=1):
    assert masks.dim() == 2, f"Only support masks.dim() == 2, but the masks.dim() == {masks.dim()}!!!"
    batch_size = masks.shape[0]
    sent_len = masks.shape[1]
    extend_masks = torch.zeros([batch_size, sent_len+extend_len], dtype=torch.bool)
    extend_masks[:, :sent_len] = masks
    return extend_masks
    
def get_batch_size(configs, batch_data):
    if isinstance(batch_data, list):
        return len(batch_data)
    elif isinstance(batch_data, torch.Tensor):
        return batch_data.shape[0]
    else:
        raise Exception("The func of get_batch_size only accepts list or torch.tensor!!!")