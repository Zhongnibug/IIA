import pickle
import os
import math
import omegaconf
import itertools
from utils.get_judge import get_caption_limit_judge

def get_struct_pair(train_gt_path, bleu_score, bleu_control, log, save_path, name="struct_pair", caption_limit_len=None, keep_self=0.0):
    obj = StructInfoPair(train_gt_path, bleu_score, bleu_control, caption_limit_len, keep_self)
    if log:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    return obj

def get_seman_pair(train_gt_path, cider_score, bleu_score, bleu_control, log, save_path, name="seman_pair", caption_limit_len=None, keep_self=0.0):
    obj = SemanticsInfoPair(train_gt_path, cider_score, bleu_score, bleu_control, caption_limit_len, keep_self)
    if log:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    return obj
    
class StructInfoPair:
    def __init__(self, train_gt_path, bleu_score, bleu_control=0, caption_limit_len=None, keep_self=0.0):

        self.sorted_caps_and_shared_segment = {}
        self.every_vid_high_caps = {}

        self.build(train_gt_path, bleu_score, bleu_control, caption_limit_len, keep_self)

    def build(self, train_gt_path, bleu_score, bleu_control=0, caption_limit_len=None, keep_self=0.0):
        with open(train_gt_path, "rb") as f:
            train_gt =  pickle.load(f)

        caption_limit_judge = get_caption_limit_judge(caption_limit_len)
        
        max_caption_len = -1
        for vid, captions in train_gt.items():
            for caption in captions:
                if caption_limit_judge(len(caption.split())):
                    if len(caption.split()) > max_caption_len:
                        max_caption_len = len(caption.split())

        for vid, captions in train_gt.items():
            vid_dic = {}
            vid_bleu_score = bleu_score.get(vid)
            caps_map = vid_bleu_score["map"]
            caps_bleus = vid_bleu_score["bleus"]
            for caption in captions:
                if caption_limit_judge(len(caption.split(" "))):
                    vid_dic[caption] = sum([l[-1] for l in caps_bleus[caps_map[caption]]])

            sorted_dic = sorted(vid_dic.items(), key=lambda item: item[1])

            if bleu_control:
                self.every_vid_high_caps[vid] = [t[0] for t in sorted_dic[-bleu_control:]]

            vid_sorted_caps_and_shared_segment = {}
            pre = sorted_dic[0][0]
            pre_lst = pre.split(" ")

            for i in range(1, len(sorted_dic)):
                cur = sorted_dic[i][0]
                cur_lst = cur.split(" ")

                sorted_dic_top = sorted_dic[-bleu_control if bleu_control else i:]

                cap_pair_bleu_score = {t[0]: tuple(caps_bleus[caps_map[pre]][caps_map[t[0]]][::-1]) for t in sorted_dic_top}
                sorted_cap_pair_bleu_score = sorted(cap_pair_bleu_score.items(), key=lambda item: item[1])
                max_sent = sorted_cap_pair_bleu_score[-1][0]
                
                all_pair_bleu_score = {t[0]: tuple(caps_bleus[caps_map[pre]][caps_map[t[0]]][::-1]) for t in sorted_dic}
                sorted_all_pair_bleu_score = sorted(all_pair_bleu_score.items(), key=lambda item: item[1])
                keey_self_pos = math.floor(len(sorted_all_pair_bleu_score) * keep_self)
                for rank, (key, value) in enumerate(sorted_all_pair_bleu_score, start=1):
                    if key == max_sent:
                        if rank < keey_self_pos:
                            max_sent = pre
                        break
                max_sent_lst = max_sent.split(" ")
                j = 0
                while(j<len(pre_lst) and j<len(max_sent_lst) and pre_lst[j]==max_sent_lst[j]):
                    j += 1
                vid_sorted_caps_and_shared_segment[pre] = ((max_sent, " ".join(pre_lst[:j])))
                pre = cur
                pre_lst = cur_lst
            vid_sorted_caps_and_shared_segment[sorted_dic[-1][0]] = ((sorted_dic[-1][0], sorted_dic[-1][0]))
            self.sorted_caps_and_shared_segment[vid] = vid_sorted_caps_and_shared_segment
    
    def get(self, vid, cap):
        return self.sorted_caps_and_shared_segment[vid][cap]
    
    def get_high_caps(self, vid):
        return self.every_vid_high_caps[vid]
    

class SemanticsInfoPair:
    def __init__(self, train_gt_path, cider_score, bleu_score, bleu_control=0, caption_limit_len=None, keep_self=0.0):

        self.sorted_caps_and_shared_segment = {}
        self.every_vid_high_caps = {}

        self.build(train_gt_path, cider_score, bleu_score, bleu_control, caption_limit_len, keep_self)

    def build(self, train_gt_path, cider_score, bleu_score, bleu_control=0, caption_limit_len=None, keep_self=0.0):
        with open(train_gt_path, "rb") as f:
            train_gt =  pickle.load(f)

        caption_limit_judge = get_caption_limit_judge(caption_limit_len)
        
        max_caption_len = -1
        for vid, captions in train_gt.items():
            for caption in captions:
                if caption_limit_judge(len(caption.split())):
                    if len(caption.split()) > max_caption_len:
                        max_caption_len = len(caption.split())

        for vid, captions in train_gt.items():
            vid_dic = {}
            vid_bleu_score = bleu_score.get(vid)
            caps_map = vid_bleu_score["map"]
            caps_bleus = vid_bleu_score["bleus"]
            for caption in captions:
                if caption_limit_judge(len(caption.split(" "))):
                    vid_dic[caption] = cider_score.get(vid, caption)

            sorted_dic = sorted(vid_dic.items(), key=lambda item: item[1])

            if bleu_control:
                self.every_vid_high_caps[vid] = [t[0] for t in sorted_dic[-bleu_control:]]

            vid_sorted_caps_and_shared_segment = {}
            pre = sorted_dic[0][0]
            pre_lst = pre.split(" ")

            for i in range(1, len(sorted_dic)):
                cur = sorted_dic[i][0]
                cur_lst = cur.split(" ")

                sorted_dic_top = sorted_dic[-bleu_control if bleu_control else i:]

                cap_pair_bleu_score = {t[0]: tuple(caps_bleus[caps_map[pre]][caps_map[t[0]]][::-1]) for t in sorted_dic_top}
                sorted_cap_pair_bleu_score = sorted(cap_pair_bleu_score.items(), key=lambda item: item[1])
                max_sent = sorted_cap_pair_bleu_score[-1][0]

                all_pair_bleu_score = {t[0]: tuple(caps_bleus[caps_map[pre]][caps_map[t[0]]][::-1]) for t in sorted_dic}
                sorted_all_pair_bleu_score = sorted(all_pair_bleu_score.items(), key=lambda item: item[1])
                keey_self_pos = math.floor(len(sorted_all_pair_bleu_score) * keep_self)
                for rank, (key, value) in enumerate(sorted_all_pair_bleu_score, start=1):
                    if key == max_sent:
                        if rank < keey_self_pos:
                            max_sent = pre
                        break

                max_sent_lst = max_sent.split(" ")

                j = 0
                while(j<len(pre_lst) and j<len(max_sent_lst) and pre_lst[j]==max_sent_lst[j]):
                    j += 1
                vid_sorted_caps_and_shared_segment[pre] = ((max_sent, " ".join(pre_lst[:j])))
                pre = cur
                pre_lst = cur_lst
            vid_sorted_caps_and_shared_segment[sorted_dic[-1][0]] = ((sorted_dic[-1][0], sorted_dic[-1][0]))
            self.sorted_caps_and_shared_segment[vid] = vid_sorted_caps_and_shared_segment
    
    def get(self, vid, cap):
        return self.sorted_caps_and_shared_segment[vid][cap]
    
    def get_high_caps(self, vid):
        return self.every_vid_high_caps[vid]