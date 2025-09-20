import pickle
import torch
from torch import nn

class LanguagePostProcess:
    def __init__(self, vocabulary_path, bos_idx, eos_idx):
        with open(vocabulary_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        pass

    def process(self, post_process_logger, videonames, outs, batch_size):
        res_list = []
        for b in range(batch_size):
            res = []
            if isinstance(outs[b], torch.Tensor):
                outs_b = outs[b].tolist()
            elif isinstance(outs[b], list):
                outs_b = outs[b]
            for x in outs_b:
                if x == self.bos_idx:
                    continue
                if x == self.eos_idx:
                    break
                res.append(x)
            res_str = self.vocab.index_list_to_string(res)
            if post_process_logger is not None:
                post_process_logger.update(f'{videonames[b]}: {res_str}')
            res_list.append(res_str)
        return res_list