import torch
from utils.get_attr import no_attr
import numpy as np
    
class seq2seq_auto_run:
    def __init__(self, muti_func_name_dict, muti_settings_dict, muti_device_dict, muti_model_args):
        self.muti_beam_search_dict = {}
        self.muti_func_name_dict = muti_func_name_dict
        self.muti_settings_dict = muti_settings_dict
        self.muti_model_args = muti_model_args
        self.muti_device_dict = muti_device_dict
        if self.muti_func_name_dict == "test_beam_search":

            self.muti_beam_search_dict["step_data_func_list"] = []
            self.muti_beam_search_dict["step_data_args_name_list"] = []
            self.muti_beam_search_dict["step_data_return_list"] = []
            for s_d_n, a_r in self.muti_settings_dict["step_data"].items():
                self.muti_beam_search_dict["step_data_func_list"].append(getattr(StepDataFunc(), s_d_n))
                self.muti_beam_search_dict["step_data_args_name_list"].append(a_r["args"])
                self.muti_beam_search_dict["step_data_return_list"].append(a_r["return"])
            
            self.muti_beam_search_dict["none_step_func_name_list"] = []
            self.muti_beam_search_dict["none_step_args_name_list"] = []
            self.muti_beam_search_dict["none_step_return_list"] = []
            if not no_attr(self.muti_model_args, "none_step"):
                for n_s, a_r in self.muti_model_args["none_step"].items():
                    self.muti_beam_search_dict["none_step_func_name_list"].append(n_s)
                    self.muti_beam_search_dict["none_step_args_name_list"].append(a_r["args"])
                    self.muti_beam_search_dict["none_step_return_list"].append(a_r["return"])

            self.muti_beam_search_dict["step_func_name_list"] = []
            self.muti_beam_search_dict["step_args_name_list"] = []
            self.muti_beam_search_dict["step_return_list"] = []
            if not no_attr(self.muti_model_args, "step"):
                for s, a_r in self.muti_model_args["step"].items():
                    self.muti_beam_search_dict["step_func_name_list"].append(s)
                    self.muti_beam_search_dict["step_args_name_list"].append(a_r["args"])
                    self.muti_beam_search_dict["step_return_list"].append(a_r["return"])
                pass
        pass

    def model_run(self, model, run_data_dict):
        return getattr(self, self.muti_func_name_dict)(model, run_data_dict)
    
    def train_one_batch_out(self, model, run_data_dict):
        model.train()
        run_data_dict[None] = None
        model_input_dict = {k: run_data_dict[v].to(self.muti_device_dict) \
                            if isinstance(run_data_dict[v], torch.Tensor) else run_data_dict[v] \
                                for k, v in self.muti_model_args.items()}
        return model(**model_input_dict)
    
    
    def test_beam_search(self, model, run_data_dict):
        
        none_step_func_list = [getattr(model, n_s_n) for n_s_n in self.muti_beam_search_dict["none_step_func_name_list"]]
        step_func_list = [getattr(model, s_n) for s_n in self.muti_beam_search_dict["step_func_name_list"]]
        run_data_dict[None] = None
        # beam search
        return beam_search(args_dict=run_data_dict, 
                           none_step_func_list=none_step_func_list, 
                           none_step_args_name_list=self.muti_beam_search_dict["none_step_args_name_list"],
                           none_step_return_list=self.muti_beam_search_dict["none_step_return_list"], 
                           repeat_args_name_dict=self.muti_settings_dict["repeat_args_name_dict"], 
                           collect_args_list=self.muti_settings_dict["collect_args_list"],
                           step_func_list=step_func_list, 
                           step_args_name_list=self.muti_beam_search_dict["step_args_name_list"], 
                           step_return_list=self.muti_beam_search_dict["step_return_list"],
                           step_data_func_list=self.muti_beam_search_dict["step_data_func_list"], 
                           step_data_args_name_list=self.muti_beam_search_dict["step_data_args_name_list"], 
                           step_data_return_list=self.muti_beam_search_dict["step_data_return_list"],
                           max_len=self.muti_settings_dict["max_step"], 
                           pad=self.muti_settings_dict["pad_idx"], 
                           bos=self.muti_settings_dict["bos_idx"], 
                           eos=self.muti_settings_dict["eos_idx"], 
                           batch_size=run_data_dict["batch_size"], 
                           beam_size=self.muti_settings_dict["beam_size"], 
                           device=self.muti_device_dict,
                           step_log_name="STEPLOG", 
                           step_dec_name="STEPDEC")

class StepDataFunc:
    '''
    The batchsize and step are fixed args.
    '''

    def tgt_attn_mask(self, batchsize, step, *args, **kwdargs):
        "Mask out subsequent positions."
        attn_shape = (step, step)
        
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

        return torch.from_numpy(subsequent_mask) == 1
    
    def seq_gen(self, batchsize, step, *args, **kwdargs):
        shape = (batchsize, step)
        seq = np.full(shape, kwdargs["mask_idx"], dtype=int)
        return torch.from_numpy(seq)

def beam_search(args_dict, none_step_func_list, none_step_args_name_list,
                none_step_return_list, repeat_args_name_dict, collect_args_list,
                step_func_list, step_args_name_list, step_return_list,
                step_data_func_list, step_data_args_name_list, step_data_return_list,
                max_len, pad, bos, eos, batch_size, beam_size, device,
                step_log_name="STEPLOG", step_dec_name="STEPDEC"):

    """ Translation work in one batch """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ Indicate the position of an instance in a tensor. """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """ Collect tensor parts associated to active instances. """

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # active instances (elements of batch) * beam search size x seq_len x h_dimension
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # select only parts of tensor which are still active
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(
            args_dict, collect_args_list, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        for k, v in args_dict.items():
            if k in collect_args_list:
                args_dict[k] = collect_active_part(v, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return args_dict, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, args_dict, 
            step_func_list, step_args_name_list,
            step_return_list, step_data_func_list,
            step_data_args_name_list, step_data_return_list,
            inst_idx_to_position_map, n_bm,
            step_log_name="STEPLOG", step_dec_name="STEPDEC"):
        """ Decode and update beam status, and then return active beam idx """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # [Batch size, Beam size, Dec Seq Len]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            batch_size = len(dec_partial_seq)
            # [Batch size* Beam size, Dec Seq Len]
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq, batch_size

        def predict_word(args_dict, step_func_list, step_args_name_list,
                         step_return_list, step_data_func_list,
                         step_data_args_name_list, step_data_return_list,
                         n_active_inst, len_dec_seq, batch_size, n_bm,
                         step_log_name = "STEPLOG"):
            
            for i, s_d_func in enumerate(step_data_func_list):
                step_data = s_d_func(batch_size*n_bm, len_dec_seq,
                                     **{k: args_dict[v] if isinstance(v, str) else v for k, v in step_data_args_name_list[i].items()})
                if isinstance(step_data, tuple):
                    for j, return_name in enumerate(step_data_return_list[i]):
                        args_dict[return_name] = step_data[j].to(device) if isinstance(step_data[j], torch.Tensor) else step_data[j]
                else:
                    args_dict[step_data_return_list[i][0]] = step_data.to(device) if isinstance(step_data, torch.Tensor) else step_data

            for i, s_func in enumerate(step_func_list):
                dec = s_func(**{k: args_dict[v] if isinstance(v, str) else v for k, v in step_args_name_list[i].items()})
                if isinstance(dec, tuple):
                    for j, return_name in enumerate(step_return_list[i]):
                        args_dict[return_name] = dec[j]
                else:
                    args_dict[step_return_list[i][0]] = dec
            
            # (active_bs, beansize, vocab)
            word_logprob = args_dict[step_log_name][:, -1].view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])  # Fill Beam object with assigned probabilities
                if not is_inst_complete:  # if top beam ended with eos, we do not add it
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # get decoding sequence for each beam
        # [Batch size* Beam size, Dec Seq Len]
        args_dict[step_dec_name], batch_size = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)


        # get word probabilities for each beam
        # size: Batch size x Beam size x Vocabulary
        word_logprob = predict_word(args_dict, step_func_list, step_args_name_list,
                                    step_return_list, step_data_func_list,
                                    step_data_args_name_list, step_data_return_list,
                                    n_active_inst, len_dec_seq, batch_size, n_bm,
                                    step_log_name=step_log_name)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores
    
    # Repeat data
    def repeat_data(a, bm_n):
        bs, *tail_shape = a.size()
        repeats = [1] * len(a.shape)
        repeats[1] = bm_n
        return a.repeat(*repeats).view(bs * bm_n, *tail_shape) 

    # args_dict

    # repeat_args_name_dict
    # because maybe some args will have repeated and non-repeated state
    # but they will update args_dict

    # none_step_func_list
    # none_step_args_name_list
    # none_step_return_list

    # repeat_args_name_dict
    # because maybe some args will have repeated and non-repeated state
    # but they will update args_dict
    
    # step_func_list
    # step_args_name_list
    # step_return_list

    # step_data_func_name_list
    # step_data_args_name_list(fixed batchsize=(original batchsize*beamsize), step)
    # step_data_return_list

    # collect_args_list

    # order: non-step, repeat, circle(step_data, step, collect_args)

    with torch.no_grad():

        # to device
        for k, v in args_dict.items():
            args_dict[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        # -- Encode
        for i, n_s_func in enumerate(none_step_func_list):
            src_enc = n_s_func(**{k: args_dict[v] for k, v in none_step_args_name_list[i].items()})
            if isinstance(src_enc, tuple):
                for j, return_name in enumerate(none_step_return_list[i]):
                    args_dict[return_name] = src_enc[j]
            else:
                args_dict[none_step_return_list[i][0]] = src_enc

        # Repeat data for beam search
        for repeated_name, repeat_name in repeat_args_name_dict.items():
            args_dict[repeated_name] = repeat_data(args_dict[repeat_name], beam_size)

        # -- Prepare beams
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_len + 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, args_dict, 
                step_func_list, step_args_name_list,
                step_return_list, step_data_func_list,
                step_data_args_name_list, step_data_return_list,
                inst_idx_to_position_map, beam_size,
                step_log_name=step_log_name, step_dec_name=step_dec_name
                )

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            # filter out inactive tensor parts (for already decoded sequences)
            args_dict, inst_idx_to_position_map = collate_active_info(
                args_dict, collect_args_list, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, beam_size)

    return [b[0] for b in batch_hyp]

class Beam:
    """ Beam search """

    def __init__(self, size, pad, bos, eos, device=False):

        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        # Initialize to [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        """Update beam status and check if finished or not."""
        num_words = word_logprob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # in initial case,
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)


        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        # print(k.type())
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

