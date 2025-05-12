from utils.get_obj import get_obj_from_str
import pickle
import torch
from collections import OrderedDict, defaultdict
from evaluate.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from evaluate.coco_caption.pycocoevalcap.cider.cider import Cider
from evaluate.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from evaluate.coco_caption.pycocoevalcap.rouge.rouge import Rouge
import psutil

class LanguageEval:
    def __init__(self, eval_class_list=["Bleu", "Cider", "Meteor", "Rouge"]):
        self.eval_class_list = eval_class_list
        self.sample_seqs = []
        self.groundtruth_seqs = []
        pass

    def clear(self):
        self.sample_seqs = []
        self.groundtruth_seqs = []
        
    def collect_data(self, predictions, gts):
        self.sample_seqs += predictions
        self.groundtruth_seqs += gts

    def eval_summary(self, eval_logger):
        assert len(self.sample_seqs) == len(self.groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'
        eval_class_list_lower = [eval_class.lower() for eval_class in self.eval_class_list]
        references, predictions = OrderedDict(), OrderedDict()

        for i in range(len(self.groundtruth_seqs)):
            references[i] = [self.groundtruth_seqs[i][j] for j in range(len(self.groundtruth_seqs[i]))]
        for i in range(len(self.sample_seqs)):
            predictions[i] = [self.sample_seqs[i]]

        predictions = {i: predictions[i] for i in range(len(self.sample_seqs))}
        references = {i: references[i] for i in range(len(self.groundtruth_seqs))}

        final_output = {}
        
        if "bleu" in eval_class_list_lower or \
            "bleu_1" in eval_class_list_lower or \
                "bleu_2" in eval_class_list_lower or \
                    "bleu_3" in eval_class_list_lower or \
                        "bleu_4" in eval_class_list_lower or\
                            "sum" in eval_class_list_lower:
            avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
            if "bleu" in eval_class_list_lower:
                eval_logger.update(f'avg_bleu_score == {[k * 100 for k in avg_bleu_score]}')
                final_output['BLEU'] = [k * 100 for k in avg_bleu_score]
            if "bleu_1" in eval_class_list_lower:
                eval_logger.update(f'avg_bleu_1_score == {avg_bleu_score[0] * 100}')
                final_output['BLEU_1'] = avg_bleu_score[0] * 100
                eval_logger.output_info("bleu_1", bleu_score[0])
            if "bleu_2" in eval_class_list_lower:
                eval_logger.update(f'avg_bleu_2_score == {avg_bleu_score[1] * 100}')
                final_output['BLEU_2'] = avg_bleu_score[1] * 100
            if "bleu_3" in eval_class_list_lower:
                eval_logger.update(f'avg_bleu_3_score == {avg_bleu_score[2] * 100}')
                final_output['BLEU_3'] = avg_bleu_score[2] * 100
            if "bleu_4" in eval_class_list_lower:
                eval_logger.update(f'avg_bleu_4_score == {avg_bleu_score[3] * 100}')
                final_output['BLEU_4'] = avg_bleu_score[3] * 100
                eval_logger.output_info("bleu_4", bleu_score[3])

        if "cider" in eval_class_list_lower or "sum" in eval_class_list_lower: 
            avg_cider_score, cider_score = Cider().compute_score(references, predictions)
            if "cider" in eval_class_list_lower:
                eval_logger.update(f'avg_cider_score == {avg_cider_score * 100}')
                final_output['CIDEr'] = avg_cider_score * 100
                eval_logger.output_info("cider", cider_score)

        if "meteor" in eval_class_list_lower or "sum" in eval_class_list_lower:
            meteor_instance = Meteor()
            avg_meteor_score, meteor_score = meteor_instance.compute_score(references, predictions)
            meteor_instance.__exit__()
            if "meteor" in eval_class_list_lower:
                eval_logger.update(f'avg_meteor_score == {avg_meteor_score * 100}')
                final_output['METEOR'] = avg_meteor_score * 100

        if "rouge" in eval_class_list_lower or "sum" in eval_class_list_lower:
            avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
            if "rouge" in eval_class_list_lower:
                eval_logger.update(f'avg_rouge_score == {avg_rouge_score * 100}')
                final_output['ROUGE'] = avg_rouge_score * 100

        if "sum" in eval_class_list_lower:
            sum_score = (avg_bleu_score[3]+avg_cider_score+avg_meteor_score+avg_rouge_score) * 100
            eval_logger.update(f'sum_score == {sum_score}')
            final_output['Sum'] = sum_score

        # length info
        if "length" in eval_class_list_lower:
            length_sum = 0
            for seq in self.sample_seqs:
                length_sum += len(seq.split(' '))
            eval_logger.update(f'avg_length == {length_sum /len(self.sample_seqs)}')
            final_output['Length'] = length_sum /len(self.sample_seqs)

        return final_output
    
class LanguageAndInfoEval:
    def __init__(self, vocab_path, info_freq_path, 
                 eval_class_list=["Bleu", "Cider", "Meteor", "Rouge"]):
        self.eval_class_list = eval_class_list
        self.sample_seqs = []
        self.groundtruth_seqs = []
        self.language_eval = LanguageEval(self.eval_class_list)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        with open(info_freq_path, "rb") as f:
            self.info_freq = pickle.load(f)
        
    def clear(self):
        self.sample_seqs = []
        self.groundtruth_seqs = []
        
    def collect_data(self, predictions, gts):
        self.sample_seqs += predictions
        self.groundtruth_seqs += gts

    def eval_summary(self, eval_logger):
        self.language_eval.sample_seqs = self.sample_seqs
        self.language_eval.groundtruth_seqs = self.groundtruth_seqs
        final_output = self.language_eval.eval_summary(eval_logger)
        eval_class_list_lower = [eval_class.lower() for eval_class in self.eval_class_list]
        
        # info level
        if "info_level" in eval_class_list_lower:
            info_level_sum = 0
            for seq in self.sample_seqs:
                info_level_sum += self.info_freq.get(self.vocab.string_to_index_list(seq))
            eval_logger.update(f'avg_info_level == {info_level_sum /len(self.sample_seqs)}')
            final_output['Info_Level'] = info_level_sum /len(self.sample_seqs)

        return final_output