import torch
import subwork
import numpy as np

def auto_collate_function(batch, configs, non_dataset_data, valid_keys,
                          function_list, input_keys_list, output_keys_list, use_list):
    
    dataloaddict = tuple(zip(*batch))
    dataloaddict = {v: dataloaddict[i] for i,v in enumerate(valid_keys)}
    dataloaddict.update(non_dataset_data)
    
    for i in range(len(function_list)):
        output = function_list[i](configs, **{k:dataloaddict[v]for k,v in input_keys_list[i].items()})
        if isinstance(output, tuple):
            for j, outputkey in enumerate(output_keys_list[i]):
                dataloaddict[outputkey] = output[j]
        else:
            dataloaddict[output_keys_list[i][0]] = output

    return tuple((dataloaddict[key] for key in use_list))


def easy_collate_function_val_test_video_caption(batch, configs, *args):
    features, groundtruths, videonames = zip(*batch)
    batch_size = len(features)
    feature_shape = list(features[0].shape[1:])

    longest_features_len = max([item.shape[0] for item in features])

    ret_features_shape = [batch_size, longest_features_len] + feature_shape
    ret_features =  torch.zeros(ret_features_shape, dtype=torch.float)
    features_masks = torch.ones([batch_size, longest_features_len], dtype=torch.bool)

    for i in range(batch_size):
        ret_features[i, :features[i].shape[0]] = features[i]
        features_masks[i, :features[i].shape[0]] = False

    groundtruths = list(groundtruths)
    videonames = list(videonames)

    return ret_features, features_masks, \
            groundtruths, videonames

def random_noise_high_frequence_collate_function_train_video_caption(batch, configs, frequence, *args):
    captions, labels, valid_lens, \
    features, groundtruths, videonames = zip(*batch)

    batch_size = len(captions)
    feature_shape = list(features[0].shape[1:])
    valid_lens = list(valid_lens)
    longest_captions_len = max(valid_lens)
    longest_features_len = max([item.shape[0] for item in features])

    ret_captions_shape = [batch_size, longest_captions_len]
    ret_captions = torch.LongTensor(*ret_captions_shape).fill_(configs.pad_idx)
    ret_labels = torch.LongTensor(*ret_captions_shape).fill_(configs.pad_idx)

    # captions_masks = subsequent_mask_3d(batch_size, longest_captions_len)
    # using 2d is ok!
    captions_masks = subsequent_mask_2d(longest_captions_len)

    ret_features_shape = [batch_size, longest_features_len] + feature_shape
    ret_features =  torch.zeros(ret_features_shape, dtype=torch.float)
    features_masks = torch.ones([batch_size, longest_features_len], dtype=torch.bool)

    frequence_masks = torch.ones([batch_size, longest_captions_len], dtype=torch.bool)

    groundtruths = list(groundtruths)
    videonames = list(videonames)

    for i in range(batch_size):
        ret_captions[i, :captions[i].shape[0]] = captions[i]
        ret_labels[i, :captions[i].shape[0]] = labels[i]

        ret_features[i, :features[i].shape[0]] = features[i]
        features_masks[i, :features[i].shape[0]] = False
        for j in range(captions[i].shape[0]):
            frequence_masks[i][j] = frequence.get(videonames[i],captions[i][j].item())

    return ret_captions, ret_labels, captions_masks, valid_lens, \
            ret_features, features_masks, \
            groundtruths, videonames, frequence_masks


def frequence_collate_function_train_video_caption(batch, configs, frequence, *args):
    captions, labels, valid_lens, \
    features, groundtruths, videonames = zip(*batch)

    batch_size = len(captions)
    feature_shape = list(features[0].shape[1:])
    valid_lens = list(valid_lens)
    longest_captions_len = max(valid_lens)
    longest_features_len = max([item.shape[0] for item in features])

    ret_captions_shape = [batch_size, longest_captions_len]
    ret_captions = torch.LongTensor(*ret_captions_shape).fill_(configs.pad_idx)
    ret_labels = torch.LongTensor(*ret_captions_shape).fill_(configs.pad_idx)

    # captions_masks = subsequent_mask_3d(batch_size, longest_captions_len)
    # using 2d is ok!
    captions_masks = subsequent_mask_2d(longest_captions_len)

    ret_features_shape = [batch_size, longest_features_len] + feature_shape
    ret_features =  torch.zeros(ret_features_shape, dtype=torch.float)
    features_masks = torch.ones([batch_size, longest_features_len], dtype=torch.bool)

    low_freq_sets_dict = {}
    high_freq_sets_dict = {}

    common_low_freq_set = set()
    common_high_freq_set = set()

    groundtruths = list(groundtruths)
    videonames = list(videonames)

    for i in range(batch_size):
        ret_captions[i, :captions[i].shape[0]] = captions[i]
        ret_labels[i, :captions[i].shape[0]] = labels[i]

        ret_features[i, :features[i].shape[0]] = features[i]
        features_masks[i, :features[i].shape[0]] = False
        low_freq_sets_dict[i] = frequence.get_low_freq_set(videonames[i])
        high_freq_sets_dict[i] = frequence.get_high_freq_set(videonames[i])

        common_low_freq_set |= low_freq_sets_dict[i]
        common_high_freq_set |= high_freq_sets_dict[i]
    
    common_low_freq_tensor = torch.tensor(list(common_low_freq_set))
    common_high_freq_tensor = torch.tensor(list(common_high_freq_set))

    common_low_freq_reverse = torch.zeros([torch.max(common_low_freq_tensor)+1], dtype=torch.long)
    common_high_freq_reverse = torch.zeros([torch.max(common_high_freq_tensor)+1],dtype=torch.long)

    low_freq_masks = torch.ones([batch_size, common_low_freq_tensor.shape[0]], dtype=torch.bool)
    high_freq_masks = torch.ones([batch_size, common_high_freq_tensor.shape[0]], dtype=torch.bool)

    for i in range(batch_size):
        low_freq_masks[i, common_low_freq_reverse[torch.tensor(list(low_freq_sets_dict[i]))]] = False
        high_freq_masks[i, common_high_freq_reverse[torch.tensor(list(high_freq_sets_dict[i]))]] = False

    reverse_high_freqs = torch.full([batch_size, configs.CheckConfigs.model.args.tgt_vocab], -1)
    reverse_high_freqs[torch.arange(batch_size).repeat_interleave(common_high_freq_tensor.shape[0]), common_high_freq_tensor.repeat(batch_size)] = \
        torch.arange(common_high_freq_tensor.shape[0]).repeat(batch_size) * (~high_freq_masks.view(-1)) - high_freq_masks.view(-1) * 1

    return ret_captions, ret_labels, captions_masks, valid_lens, \
            ret_features, features_masks, \
            groundtruths, videonames, \
            reverse_high_freqs, \
            common_low_freq_tensor, common_high_freq_tensor, \
            low_freq_masks, high_freq_masks

def subsequent_mask_2d(configs, size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 1