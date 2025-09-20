import pickle
import sys
import omegaconf
import h5py
import numpy as np
import torch
from utils.get_attr import no_attr
from utils.get_judge import get_caption_limit_judge

def open_and_load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data =  pickle.load(f)
    return data
    
def open_and_read_h5_according_keys_equidistant(h5_path, keys, nums, main_key=None, child_key=None):
    
    data = {}
    data_masks = {}

    with h5py.File(h5_path, 'r') as f:

        if main_key is None:
            key_data = f
        else:
            main_key_list = main_key.split('.')
            key_data = f
            for k in main_key_list:
                key_data = key_data[k]

        for key in keys:
            if child_key is None:
                temp_data = key_data[key][()]
            else:
                temp_data = key_data[key][child_key][()]

            if len(temp_data)>=nums:
                sampled_idxs = np.linspace(0, len(temp_data) - 1, nums, dtype=int)
                data[key] = temp_data[sampled_idxs]
                data_masks[key] = np.zeros(nums, dtype=bool)
            else:
                feat_shape = temp_data.shape
                new_data = np.full((nums, *feat_shape[1:]), fill_value=0.0)
                new_data[:feat_shape[0]] = temp_data[:]
                data[key] = new_data
                mask = np.ones(nums, dtype=bool)
                mask[:feat_shape[0]] = False
                data_masks[key] = mask
    return data, data_masks

def open_and_read_h5_according_keys_index_equidistant(h5_path, keys, nums, main_key=None):
    
    data = {}
    data_masks = {}

    with h5py.File(h5_path, 'r') as f:

        if main_key is None:
            key_data = f
        else:
            main_key_list = main_key.split('.')
            key_data = f
            for k in main_key_list:
                key_data = key_data[k]

        all_key_data = key_data[()]
        for key_index in range(len(keys)):
            if len(all_key_data[0])>=nums:
                sampled_idxs = np.linspace(0, len(all_key_data[0]) - 1, nums, dtype=int)
                data[keys[key_index]] = all_key_data[key_index][sampled_idxs]
                data_masks[keys[key_index]] = np.zeros(nums, dtype=bool)
            else:
                feat_shape = all_key_data[key_index].shape
                new_data = np.full((nums, *feat_shape[1:]), fill_value=0.0)
                new_data[:len(feat_shape[0])] = all_key_data[key_index][:]
                data[keys[key_index]] = new_data
                mask = np.ones(nums, dtype=bool)
                mask[:feat_shape[0]] = False
                data_masks[keys[key_index]] = mask

    return data, data_masks

def caption_find_videoname_limit_len(groundtruths, vocabulary, bos_idx, eos_idx, pad_idx, caption_limit_len=None):

    caption_limit_judge = get_caption_limit_judge(caption_limit_len)

    numberic_caps = []
    numberic_labels = []
    valid_lens = []
    videonames = []

    max_caption_len = -1
    for vid, captions in groundtruths.items():
        for caption in captions:
            if caption_limit_judge(len(caption.split())):
                if len(caption.split()) > max_caption_len:
                    max_caption_len = len(caption.split())

    for vid, captions in groundtruths.items():
        for caption in captions:
            if caption_limit_judge(len(caption.split())):
                numberic_cap = vocabulary.string_to_index_list(caption)
                numberic_caps.append([bos_idx]+numberic_cap+[pad_idx]*(max_caption_len-len(caption.split())))
                numberic_labels.append(numberic_cap+[eos_idx]+[pad_idx]*(max_caption_len-len(caption.split())))
                valid_lens.append(len(numberic_cap)+1)
                videonames.append(vid)
        

    split_names = list(groundtruths.keys())
    return numberic_caps, \
            numberic_labels, \
            valid_lens, \
            videonames, \
            len(numberic_caps), \
            split_names

def caption_with_pair(groundtruths, vocabulary, pair,
                      bos_idx, eos_idx, pad_idx,
                      caption_limit_len=None):

    caption_limit_judge = get_caption_limit_judge(caption_limit_len)

    low_numberic_caps = []
    low_numberic_labels = []
    low_valid_lens = []

    high_numberic_caps = []
    high_numberic_labels = []
    high_valid_lens = []


    videonames = []

    max_caption_len = -1
    for vid, captions in groundtruths.items():
        for caption in captions:
            if caption_limit_judge(len(caption.split())):
                if len(caption.split()) > max_caption_len:
                    max_caption_len = len(caption.split())

    for vid, captions in groundtruths.items():
        for caption in captions:
            if caption_limit_judge(len(caption.split())):
                high_cap, shared_seg = pair.get(vid, caption)

                low_numberic_cap = vocabulary.string_to_index_list(caption)
                low_numberic_caps.append([bos_idx]+low_numberic_cap+[pad_idx]*(max_caption_len-len(low_numberic_cap)))
                low_numberic_labels.append(low_numberic_cap+[eos_idx]+[pad_idx]*(max_caption_len-len(low_numberic_cap)))
                low_valid_lens.append(len(low_numberic_cap)+1)


                high_numberic_cap = vocabulary.string_to_index_list(high_cap)
                high_numberic_caps.append([bos_idx]+high_numberic_cap+[pad_idx]*(max_caption_len-len(high_numberic_cap)))
                high_numberic_labels.append(high_numberic_cap+[eos_idx]+[pad_idx]*(max_caption_len-len(high_numberic_cap)))
                high_valid_lens.append(len(high_numberic_cap)+1)

                videonames.append(vid)

    split_names = list(groundtruths.keys())
    return low_numberic_caps, \
            low_numberic_labels, \
            low_valid_lens, \
            high_numberic_caps, \
            high_numberic_labels, \
            high_valid_lens, \
            videonames, \
            len(low_numberic_caps), \
            split_names

def repeat_tensor_like(muti_tensor_like, repeat_num):
    if isinstance(muti_tensor_like, dict):
        res = {}
        if isinstance(next(iter(muti_tensor_like.values())), torch.Tensor):
            for k, t in muti_tensor_like.items():
                res[k] = t.repeat(repeat_num)
        elif isinstance(next(iter(muti_tensor_like.values())), np.ndarray):
            for k, t in muti_tensor_like.items():
                res[k] = np.tile(t, repeat_num)
        else:
            raise Exception("Unkown type!!!")
    elif isinstance(muti_tensor_like, list):
        res = []
        if isinstance(muti_tensor_like[0], torch.Tensor):
            for t in muti_tensor_like:
                res.append(t.repeat(repeat_num))
        elif isinstance(muti_tensor_like[0], np.ndarray):
            for t in muti_tensor_like:
                res.append(np.tile(t, repeat_num))
        else:
            raise Exception("Unkown type!!!")
    else:
        raise Exception("Only process dict or list!!!")
    return res

def masks_extend(masks, extend_len=1):
    extend_masks = {}

    for vid, mask in masks.items():
        mask_shape = mask.shape
        extend_mask = np.zeros((mask_shape[0]+extend_len, *mask_shape[1:]), dtype=bool)
        extend_mask[:mask_shape[0]] = mask
        extend_masks[vid] = extend_mask

    return extend_masks

def caption_get_name_list(groundtruths):
    videonames = list(groundtruths.keys())
    split_names = list(groundtruths.keys())
    return videonames, len(groundtruths), split_names
