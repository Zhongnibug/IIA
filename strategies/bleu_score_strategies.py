import pickle
import os
from utils.get_judge import get_caption_limit_judge
from evaluate.coco_caption.pycocoevalcap.bleu.bleu import Bleu

def get_saved(saved_folder_path, name, log=False, save_path=None, is_backup=False):
    path = os.path.join(saved_folder_path, f"{name}.pkl")
    with open(path, "rb") as f:
        obj =  pickle.load(f)
    if log and save_path is not None and is_backup:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)                
    return obj

def get_bleu_score(train_gt_path, log, save_path, name="bleu_score", caption_limit_len=None, none_repeat=False):
    obj = BleuScore(train_gt_path, caption_limit_len, none_repeat)
    if log:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    return obj

class BleuScore:
    def __init__(self, train_gt_path, caption_limit_len=None, none_repeat=False):
        
        self.all_vids_caps_bleus = {}
        self.build(train_gt_path, caption_limit_len, none_repeat)

    def build(self, train_gt_path, caption_limit_len=None, none_repeat=False):
        with open(train_gt_path, "rb") as f:
            train_gt =  pickle.load(f)

        caption_limit_judge = get_caption_limit_judge(caption_limit_len)

        for vid, _ in train_gt.items():
            self.all_vids_caps_bleus[vid] = {}

        for vid, captions in train_gt.items():
            caps_lst = []
            for caption in captions:
                if caption_limit_judge(len(caption.split(" "))) and not (none_repeat and caption in caps_lst):
                    caps_lst.append(caption)

            caps_map = {}
            caps_bleus = []
            for i in range(len(caps_lst)):
                cap_bleus = []
                for j in range(len(caps_lst)):
                    avg_bleu_score, bleu_score = Bleu(4).compute_score({0: [caps_lst[j]]}, {0: [caps_lst[i]]})
                    cap_bleus.append(avg_bleu_score)
                caps_bleus.append(cap_bleus)
                caps_map[caps_lst[i]] = i

            self.all_vids_caps_bleus[vid]["map"] = caps_map
            self.all_vids_caps_bleus[vid]["bleus"] = caps_bleus
    
    def get(self, vid):
        return self.all_vids_caps_bleus[vid]