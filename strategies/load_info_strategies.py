import pickle
import os
import omegaconf
from omegaconf import OmegaConf
from subwork import subwork

def load_pkl(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file

def get_data_info_for_video_caption(configs):
    with open(configs.captions.vocabulary, "rb") as f:
        vocab = pickle.load(f)
        vocab_num = vocab.length
        pad_idx = vocab.w2i[configs.captions.pad]
        unk_idx = vocab.w2i[configs.captions.unk]
        bos_idx = vocab.w2i[configs.captions.bos]
        eos_idx = vocab.w2i[configs.captions.eos]
    return vocab, vocab_num, pad_idx, unk_idx, bos_idx, eos_idx

def get_test_model_weights_path_from_best_epoch(configs):
    work_detail_path = os.path.join(configs.test_work.work_path, "workdetail.yaml")
    work_detail = OmegaConf.load(work_detail_path)
    best_epoch = work_detail.WORKDETAIL.best_epoch
    model_weights_path = os.path.join(configs.test_work.work_path, "checkpoint", f"model_epoch{best_epoch:04d}.pth")
    return model_weights_path

def load_or_build(load_task, build_task, path, name, configs, workdata, name_suffix="pkl"):
    work_detail = {}
    if os.path.exists(os.path.join(path, f"{name}.{name_suffix}")):
        func_name = subwork({workdata["self_word"][0]: configs, 
                                workdata["workdata_word"][0]: workdata},
                                load_task,
                                f"{name}_load",
                                self_dic=configs,
                                work_detail=work_detail,
                                detail_word=workdata["detail_word"][0])

    else:
        func_name = subwork({workdata["self_word"][0]: configs, 
                                workdata["workdata_word"][0]: workdata},
                                build_task,
                                f"{name}_build",
                                self_dic=configs,
                                work_detail=work_detail,
                                detail_word=workdata["detail_word"][0])
    return work_detail