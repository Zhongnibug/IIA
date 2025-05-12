import pickle
import os
import math
import omegaconf
import itertools
from utils.get_judge import get_caption_limit_judge
from collections import OrderedDict, defaultdict
from evaluate.coco_caption.pycocoevalcap.cider.cider import Cider

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

def get_cider_score(train_gt_path, log, save_path, name="cider_score", caption_limit_len=None, none_repeat=False):
    obj = CiderScore(train_gt_path, caption_limit_len, none_repeat)
    if log:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    return obj

class CiderScore:
    def __init__(self, train_gt_path, caption_limit_len=None, none_repeat=False):
        
        self.all_vids_caps_ciders = {}
        self.build(train_gt_path, caption_limit_len, none_repeat)

    def build(self, train_gt_path, caption_limit_len=None, none_repeat=False):
        with open(train_gt_path, "rb") as f:
            train_gt =  pickle.load(f)

        caption_limit_judge = get_caption_limit_judge(caption_limit_len)
        
        vid_names = []
        vid_preds = []
        vid_gts = []
        for vid, captions in train_gt.items():
            vid_none_repeat_captions = []
            for caption in captions:
                if caption_limit_judge(len(caption.split(" "))) and not (none_repeat and caption in vid_none_repeat_captions):
                    vid_none_repeat_captions.append(caption)
                    vid_names.append(vid)
                    vid_preds.append(caption)
                    vid_gts.append(captions)

        references, predictions = OrderedDict(), OrderedDict()

        for i in range(len(vid_gts)):
            references[i] = [vid_gts[i][j] for j in range(len(vid_gts[i]))]
        for i in range(len(vid_preds)):
            predictions[i] = [vid_preds[i]]

        predictions = {i: predictions[i] for i in range(len(vid_preds))}
        references = {i: references[i] for i in range(len(vid_gts))}
        
        avg_cider_score, cider_score = Cider().compute_score(references, predictions)

        for vid, _ in train_gt.items():
            self.all_vids_caps_ciders[vid] = {}
            
        for i in range(len(cider_score)):
            self.all_vids_caps_ciders[vid_names[i]][vid_preds[i]] = cider_score[i]
    
    def get(self, vid, cap):
        return self.all_vids_caps_ciders[vid][cap]