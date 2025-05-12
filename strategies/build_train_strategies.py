from utils.get_obj import get_obj_from_str
from utils.get_attr import no_attr, convert_all_f64_to_f32
import torch
import pickle
import os
import omegaconf
from omegaconf import OmegaConf
import time
import psutil
from work import work

class BaseTrainer:
    def __init__(self, model, loss_compute, epochs, save_path, device=None, log=True, is_print=True, group=False):
        self.loss_compute = loss_compute
        self.epochs = epochs
        self.save_path = save_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = model.to(self.device)
        self.log = log
        self.is_print = is_print
        self.group = group
        pass

    def train():
        pass

def auto_train(model, save_path, log, workdata,
               train_cfg, train_dataloader, train_use_list,
               is_print=True, group=False,
               val_cfg=None, val_dataloader=None, val_use_list=None,
               test_cfg=None, test_dataloader=None, test_use_list=None,
               inherit_cfg=None, group_cfg=None):

    trainer = AutoTrainer(model, save_path, workdata, log,
                           train_cfg, val_cfg, test_cfg, inherit_cfg,
                           is_print, group, group_cfg)
    return trainer.train(train_dataloader, train_use_list,
                         val_dataloader, val_use_list,
                         test_dataloader, test_use_list)
    

class AutoTrainer(BaseTrainer):
    def __init__(self, model, save_path, workdata, log, 
                 train_cfg, val_cfg=None, test_cfg=None, inherit_cfg=None,
                 is_print=True, group=False, group_cfg=None):

        epochs = train_cfg.epochs
        if no_attr(train_cfg, "device"):
            device = None
        else:
            device = train_cfg.device

        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg

        self.inherit_cfg = inherit_cfg
        self.group_cfg = group_cfg
        self.workdata = workdata     

        criterion = get_obj_from_str(
            train_cfg.loss_compute.criterion.criterion_strategies
            )(**train_cfg.loss_compute.criterion.args)
        optimizer = get_obj_from_str(
            train_cfg.loss_compute.optimizer.optimizer_strategies
        )(
            model=model,
            **train_cfg.loss_compute.optimizer.args
        )
        # scheduler
        if self.train_cfg is None or (no_attr(self.train_cfg, "loss_compute")) or \
            (no_attr(self.train_cfg.loss_compute, "scheduler")) or \
                (no_attr(self.train_cfg.loss_compute.scheduler, "scheduler_strategies")):
            self.scheduler = None
        elif no_attr(self.train_cfg.loss_compute.scheduler, "args"):
            self.scheduler = get_obj_from_str(train_cfg.loss_compute.scheduler.scheduler_strategies)(optimizer=optimizer.optimizer)
        else:
            self.scheduler = get_obj_from_str(train_cfg.loss_compute.scheduler.scheduler_strategies)(optimizer=optimizer.optimizer, **self.train_cfg.loss_compute.scheduler.args)

        # loss compute settings
        self.loss_compute_out = self.train_cfg.loss_compute.loss_compute_out

        if self.train_cfg is None or (no_attr(self.train_cfg, "loss_compute")) or \
            (no_attr(self.train_cfg.loss_compute, "init_args")):
            self.loss_compute_init_args = {}
        else:
            self.loss_compute_init_args = self.train_cfg.loss_compute.init_args
        self.loss_compute_args =  self.train_cfg.loss_compute.loss_compute_args
        loss_compute = get_obj_from_str(
            train_cfg.loss_compute.loss_compute_strategies
        )(device=device, criterion=criterion, optimizer=optimizer, **self.loss_compute_init_args)

        super(AutoTrainer, self).__init__(model, loss_compute, epochs, save_path, device, log, is_print, group)

        # model run
        self.model_args_train = self.train_cfg.model_run.model_args
        if self.test_cfg is None or (no_attr(self.test_cfg, "model_run")):
            self.model_args_test = None
        else:
            self.model_args_test = self.test_cfg.model_run.model_args
        
        if self.val_cfg is None or (no_attr(self.val_cfg, "model_run")):
            self.model_args_val = self.model_args_test
        else:
            self.model_args_val = self.val_cfg.model_run.model_args

        if self.train_cfg is None or (no_attr(self.train_cfg, "model_run")):
            self.model_settings_train = {}
        else:
            self.model_settings_train = self.train_cfg.model_run.model_settings
        if self.test_cfg is None or \
            (no_attr(self.test_cfg, "model_run") or \
             (no_attr(self.test_cfg.model_run, "model_settings"))):
            self.model_settings_test = {}
        else:
            self.model_settings_test = self.test_cfg.model_run.model_settings
        
        if self.val_cfg is None or (no_attr(self.val_cfg, "model_run")):
            self.model_settings_val = self.model_settings_test
        else:
            self.model_settings_val = self.val_cfg.model_run.model_settings

        self.model_run_train = getattr(get_obj_from_str(self.train_cfg.model_run.model_run_strategies)(
                muti_func_name_dict = self.train_cfg.model_run.model_run_func,
                muti_settings_dict = self.model_settings_train,
                muti_device_dict = self.device,
                muti_model_args = self.model_args_train
        ), "model_run")
        if (self.test_cfg is None or \
            no_attr(self.test_cfg, "model_run") or \
                no_attr(self.test_cfg.model_run, "model_run_strategies")) and \
                    no_attr(self.test_cfg.model_run, "model_run_func"):
            self.model_run_strategy_test = self.train_cfg.model_run.model_run_strategies
            self.model_run_test = None
        elif (self.test_cfg is None or \
            no_attr(self.test_cfg, "model_run") or \
                no_attr(self.test_cfg.model_run, "model_run_strategies")) and \
                    not no_attr(self.test_cfg.model_run, "model_run_func"):
            self.model_run_strategy_test = self.train_cfg.model_run.model_run_strategies
            self.model_run_test = getattr(get_obj_from_str(self.model_run_strategy_test)(
                    muti_func_name_dict = self.test_cfg.model_run.model_run_func,
                    muti_settings_dict = self.model_settings_test,
                    muti_device_dict = self.device,
                    muti_model_args = self.model_args_test
            ), "model_run")
        else:
            self.model_run_strategy_test = self.test_cfg.model_run.model_run_strategies
            self.model_run_test = getattr(get_obj_from_str(self.model_run_strategy_test)(
                    muti_func_name_dict = self.test_cfg.model_run.model_run_func,
                    muti_settings_dict = self.model_settings_test,
                    muti_device_dict = self.device,
                    muti_model_args = self.model_args_test
            ), "model_run")
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "model_run") or \
                (no_attr(self.val_cfg.model_run, "model_run_strategies") and\
                 no_attr(self.val_cfg.model_run, "model_run_func")):
            self.model_run_strategy_val = self.model_run_strategy_test
            self.model_run_val = self.model_run_test
        elif self.val_cfg is None or \
            no_attr(self.val_cfg, "model_run") or \
                (no_attr(self.val_cfg.model_run, "model_run_strategies") and \
                 not no_attr(self.val_cfg.model_run, "model_run_func")):
            self.model_run_strategy_val = self.model_run_strategy_test
            self.model_run_val = getattr(get_obj_from_str(self.model_run_strategy_val)(
                    muti_func_name_dict = self.val_cfg.model_run.model_run_func,
                    muti_settings_dict = self.model_settings_val,
                    muti_device_dict = self.device,
                    muti_model_args = self.model_args_val
            ), "model_run")
        else:
            self.model_run_strategy_val = self.val_cfg.model_run.model_run_strategies
            self.model_run_val = getattr(get_obj_from_str(self.model_run_strategy_val)(
                    muti_func_name_dict = self.val_cfg.model_run.model_run_func,
                    muti_settings_dict = self.model_settings_val,
                    muti_device_dict = self.device,
                    muti_model_args = self.model_args_val
            ), "model_run")

        if self.train_cfg is None or \
            no_attr(self.train_cfg, "model_run") or \
                no_attr(self.train_cfg.model_run, "model_out"):
            self.model_out_train = None
        else:
            self.model_out_train = self.train_cfg.model_run.model_out
        if self.test_cfg is None or \
            no_attr(self.test_cfg, "model_run") or \
                no_attr(self.test_cfg.model_run, "model_out"):
            self.model_out_test = None
        else:
            self.model_out_test = self.test_cfg.model_run.model_out

        if self.val_cfg is None or \
            no_attr(self.val_cfg, "model_run") or \
                no_attr(self.val_cfg.model_run, "model_out"):
            self.model_out_val = self.model_out_test
        else:
            self.model_out_val = self.val_cfg.model_run.model_out

        # log and print
        if no_attr(self.train_cfg, "log_print_delimiter"):
            self.log_and_print_train = get_obj_from_str(self.train_cfg.log_print_strategies)(
                save_path=self.save_path,
                log=self.log
                ) 
        else:            
            self.log_and_print_train = get_obj_from_str(self.train_cfg.log_print_strategies)(
                save_path=self.save_path,
                delimiter=self.train_cfg.log_print_delimiter,
                log=self.log
                ) 
        if self.test_cfg is None or no_attr(self.test_cfg, "log_print_strategies"):
            self.log_and_print_test = self.log_and_print_train
        elif no_attr(self.test_cfg, "log_print_delimiter"):
            self.log_and_print_test = get_obj_from_str(self.test_cfg.log_print_strategies)(
                save_path=self.save_path,
                log=self.log
                )
        else:
            self.log_and_print_test = get_obj_from_str(self.test_cfg.log_print_strategies)(
                save_path=self.save_path,
                delimiter=self.test_cfg.log_print_delimiter,
                log=self.log
                )
            
        if self.val_cfg is None or no_attr(self.val_cfg, "log_print_strategies"):
            self.log_and_print_val = self.log_and_print_test
        elif no_attr(self.val_cfg, "log_print_delimiter"):
            self.log_and_print_val = get_obj_from_str(self.val_cfg.log_print_strategies)(
                save_path=self.save_path,
                log=self.log
                )            
        else:
            self.log_and_print_val = get_obj_from_str(self.val_cfg.log_print_strategies)(
                save_path=self.save_path,
                delimiter=self.val_cfg.log_print_delimiter,
                log=self.log
                )

        if (not self.is_print) or no_attr(self.train_cfg, "print") or no_attr(self.train_cfg.print, "print_freq"):
            self.print_freq_train = None
        else:
            self.print_freq_train = self.train_cfg.print.print_freq

        if (not self.is_print) or no_attr(self.train_cfg, "print") or no_attr(self.train_cfg.print, "print_summary"):
            self.print_summary_train = False
        else:
            self.print_summary_train = self.train_cfg.print.print_summary

        if no_attr(self.train_cfg, "log") or no_attr(self.train_cfg.log, "log_summary"):
            self.log_summary_train = True
        else:
            self.log_summary_train = self.train_cfg.log.log_summary

        if self.val_cfg is not None:
            if (not self.is_print) or no_attr(self.val_cfg, "print") or no_attr(self.val_cfg.print, "print_freq"):
                self.print_freq_val = None
            else:
                self.print_freq_val = self.val_cfg.print.print_freq

            if (not self.is_print) or no_attr(self.val_cfg, "print") or no_attr(self.val_cfg.print, "print_summary"):
                self.print_summary_val = False
            else:
                self.print_summary_val = self.val_cfg.print.print_summary

            if no_attr(self.val_cfg, "log") or no_attr(self.val_cfg.log, "log_summary"):
                self.log_summary_val = True
            else:
                self.log_summary_val = self.val_cfg.log.log_summary

            if (not self.is_print) or no_attr(self.val_cfg, "print") or no_attr(self.val_cfg.print, "print_post_process"):
                self.print_post_process_val = False
            else:
                self.print_post_process_val = self.val_cfg.print.print_post_process

            if no_attr(self.val_cfg, "log") or no_attr(self.val_cfg.log, "log_post_process"):
                self.log_post_process_val = True
            else:
                self.log_post_process_val = self.val_cfg.log.log_post_process

            if (not self.is_print) or no_attr(self.val_cfg, "print") or no_attr(self.val_cfg.print, "print_eval"):
                self.print_eval_val = False
            else:
                self.print_eval_val = self.val_cfg.print.print_eval

            if no_attr(self.val_cfg, "log") or no_attr(self.val_cfg.log, "log_eval"):
                self.log_eval_val = True
            else:
                self.log_eval_val = self.val_cfg.log.log_eval
        else:
            self.print_freq_val = None
            self.print_summary_val = False
            self.log_summary_val = False
            self.print_post_process_val = False
            self.log_post_process_val = False
            self.print_eval_val = False
            self.log_eval_val = False

        if self.test_cfg is not None:
            if (not self.is_print) or no_attr(self.test_cfg, "print") or no_attr(self.test_cfg.print, "print_freq"):
                self.print_freq_test = None
            else:
                self.print_freq_test = self.test_cfg.print.print_freq

            if (not self.is_print) or no_attr(self.test_cfg, "print") or no_attr(self.test_cfg.print, "print_summary"):
                self.print_summary_test = False
            else:
                self.print_summary_test = self.test_cfg.print.print_summary

            if no_attr(self.test_cfg, "log") or no_attr(self.test_cfg.log, "log_summary"):
                self.log_summary_test = True
            else:
                self.log_summary_test = self.test_cfg.log.log_summary

            if (not self.is_print) or no_attr(self.test_cfg, "print") or no_attr(self.test_cfg.print, "print_post_process"):
                self.print_post_process_test = False
            else:
                self.print_post_process_test = self.test_cfg.print.print_post_process

            if no_attr(self.test_cfg, "log") or no_attr(self.test_cfg.log, "log_post_process"):
                self.log_post_process_test = True
            else:
                self.log_post_process_test = self.test_cfg.log.log_post_process

            if (not self.is_print) or no_attr(self.test_cfg, "print") or no_attr(self.test_cfg.print, "print_eval"):
                self.print_eval_test = False
            else:
                self.print_eval_test = self.test_cfg.print.print_eval

            if no_attr(self.test_cfg, "log") or no_attr(self.test_cfg.log, "log_eval"):
                self.log_eval_test = True
            else:
                self.log_eval_test = self.test_cfg.log.log_eval
        else:
            self.print_freq_test = None
            self.print_summary_test = False
            self.log_summary_test = False

        # auto train meters params
        if no_attr(self.train_cfg, "meters") or no_attr(self.train_cfg.meters, "fmt"):
            self.train_meters_fmt = "{avg:.4f}"
        else:
            self.train_meters_fmt = self.train_cfg.meters.fmt
        if no_attr(self.train_cfg, "meters") or no_attr(self.train_cfg.meters, "num"):
            self.train_meters_num = "batch_size"
        else:
            self.train_meters_num = self.train_cfg.meters.num

        # add meters
        self.add_meters_train = {k: {"name": k, "fmt": self.train_meters_fmt} for k in self.loss_compute_out}
        if not no_attr(self.train_cfg, "meters"):
            self.add_meters_train.update({k: v["add"] for k, v in self.train_cfg.meters.items() if k!="fmt" and k!="num"})
        self.add_meters_name_list_train = [v["name"] for k, v in self.add_meters_train.items()]
        if self.test_cfg is None or no_attr(self.test_cfg, "meters"):
            self.add_meters_test = {}
            self.add_meters_name_list_test = []
        else:
            self.add_meters_test = {k: v["add"] for k, v in self.test_cfg.meters.items()}
            self.add_meters_name_list_test = [v["name"] for k, v in self.add_meters_test.items()]
        if self.val_cfg is None or no_attr(self.val_cfg, "meters"):
            self.add_meters_val = self.add_meters_test
            self.add_meters_name_list_val = self.add_meters_name_list_test
        else:
            self.add_meters_val = {k: v["add"] for k, v in self.val_cfg.meters.items()}
            self.add_meters_name_list_val = [v["name"] for _,v in self.add_meters_val.items()]

        # update meters
        self.update_meters_train = {k: {"value": k, "num": self.train_meters_num} for k in self.loss_compute_out}
        if not no_attr(self.train_cfg, "meters"):
            self.update_meters_train.update({k: v["update"] for k, v in self.train_cfg.meters.items() if k!="fmt" and k!="num"})
        if self.test_cfg is None or no_attr(self.test_cfg, "meters"):
            self.update_meters_test = {}
        else:
            self.update_meters_test = {k: v["update"] for k, v in self.test_cfg.meters.items()}
        if self.val_cfg is None or no_attr(self.val_cfg, "meters"):
            self.update_meters_val = self.update_meters_test
        else:
            self.update_meters_val = {k: v["update"] for k, v in self.val_cfg.meters.items()} 

        # Load test
        if self.test_cfg is not None:
            self.test_func_test = get_obj_from_str(self.test_cfg.build_test_strategies)
        else:
            self.test_func_test = None
        if self.val_cfg is None or no_attr(self.val_cfg, "build_test_strategies"):
            self.test_func_val = self.test_func_test
        else:
            self.test_func_val = get_obj_from_str(val_cfg.build_test_strategies)

        # End train then test
        if self.test_cfg is None or \
            no_attr(self.test_cfg, "train_end_test"):
            self.train_end_test = True
        else:
            self.train_end_test = self.test_cfg.train_end_test

        # Post process
        if self.test_cfg is None or \
            no_attr(self.test_cfg, "post_process") or \
            no_attr(self.test_cfg.post_process, "post_process_strategies"):
            self.post_process_test = None
        elif no_attr(self.test_cfg.post_process, "init_args"):
            self.post_process_test = get_obj_from_str()
        else:
            self.post_process_test = get_obj_from_str(self.test_cfg.post_process.post_process_strategies)(**self.test_cfg.post_process.init_args)
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "post_process") or \
            no_attr(self.val_cfg.post_process, "post_process_strategies"):
            self.post_process_val = self.post_process_test
        elif no_attr(self.val_cfg.post_process, "init_args"):
            self.post_process_val = get_obj_from_str()
        else:
            self.post_process_val = get_obj_from_str(self.val_cfg.post_process.post_process_strategies)(**self.val_cfg.post_process.init_args)

        if self.test_cfg is None or \
            no_attr(self.test_cfg, "post_process") or \
            no_attr(self.test_cfg.post_process, "process"):
            self.process_test = {}
        else:
            self.process_test = self.test_cfg.post_process.process
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "post_process") or \
            no_attr(self.val_cfg.post_process, "process"):
            self.process_val = self.process_test
        else:
            self.process_val = self.val_cfg.post_process.process

        if self.test_cfg is None or \
            no_attr(self.test_cfg, "post_process") or \
            no_attr(self.test_cfg.post_process, "post_process_out"):
            self.post_process_out_test = []
        else:
            self.post_process_out_test = self.test_cfg.post_process.post_process_out
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "post_process") or \
            no_attr(self.val_cfg.post_process, "post_process_out"):
            self.post_process_out_val = self.post_process_out_test
        else:
            self.post_process_out_val = self.val_cfg.post_process.post_process_out

        # eval load
        if self.test_cfg is None or \
            no_attr(self.test_cfg, "eval") or \
            no_attr(self.test_cfg.eval, "build_eval_strategies"):
            self.eval_func_test = None
        elif no_attr(self.test_cfg.eval, "init_args"):
            self.eval_func_test = get_obj_from_str(self.test_cfg.eval.build_eval_strategies)()
        else:
            self.eval_func_test = get_obj_from_str(self.test_cfg.eval.build_eval_strategies)(**self.test_cfg.eval.init_args)
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "eval") or \
            no_attr(self.val_cfg.eval, "build_eval_strategies"):
            self.eval_func_val = self.eval_func_test
        elif no_attr(self.val_cfg.eval, "init_args"):
            self.eval_func_val = get_obj_from_str(self.val_cfg.eval.build_eval_strategies)()
        else:
            self.eval_func_val = get_obj_from_str(self.val_cfg.eval.build_eval_strategies)(**self.val_cfg.eval.init_args)
        
        if self.test_cfg is None or \
            no_attr(self.test_cfg, "eval") or \
            no_attr(self.test_cfg.eval, "collect_data"):
            self.collect_data_test = {}
        else:
            self.collect_data_test = self.test_cfg.eval.collect_data
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "eval") or \
            (no_attr(self.val_cfg.eval, "collect_data") and \
             no_attr(self.val_cfg.eval, "build_eval_strategies")):
            self.collect_data_val = self.collect_data_test
        else:
            self.collect_data_val = self.val_cfg.eval.collect_data

        if self.test_cfg is None or \
            no_attr(self.test_cfg, "eval") or \
            no_attr(self.test_cfg.eval, "eval_summary"):
            self.eval_summary_test = {}
        else:
            self.eval_summary_test = self.test_cfg.eval.eval_summary
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "eval") or \
            (no_attr(self.val_cfg.eval, "eval_summary") and \
             no_attr(self.val_cfg.eval, "build_eval_strategies")):
            self.eval_summary_val = self.eval_summary_test
        else:
            self.eval_summary_val = self.val_cfg.eval.eval_summary

        if self.test_cfg is None or \
            no_attr(self.test_cfg, "eval") or \
            no_attr(self.test_cfg.eval, "eval_out"):
            self.eval_out_test = []
        else:
            self.eval_out_test = self.test_cfg.eval.eval_out
        if self.val_cfg is None or \
            no_attr(self.val_cfg, "eval") or \
            no_attr(self.val_cfg.post_process, "eval_out"):
            self.eval_out_val = self.eval_out_test
        else:
            self.eval_out_val = self.val_cfg.eval.eval_out

        # control stop
        if no_attr(self.train_cfg, "control_stop") or \
            no_attr(self.train_cfg.control_stop, "build_control_stop_strategies"):
            self.control_stop_func = None
        elif no_attr(self.train_cfg.control_stop, "init_args"):
            self.control_stop_func = get_obj_from_str(self.train_cfg.control_stop.build_control_stop_strategies)()
        else:
            self.control_stop_func = get_obj_from_str(
                self.train_cfg.control_stop.build_control_stop_strategies
                )(**self.train_cfg.control_stop.init_args)
            
        # model save
        if log:
            self.checkpoint_folder_path = os.path.join(self.save_path, "checkpoint")
            if not os.path.exists(self.checkpoint_folder_path):
                os.makedirs(self.checkpoint_folder_path)
        else:
            self.checkpoint_folder_path = None

        # only save best and last
        if no_attr(self.train_cfg, "save_only_best_and_last_ckpt"):
            self.save_only_best_and_last_ckpt = True
        else:
            self.save_only_best_and_last_ckpt = self.train_cfg.save_only_best_and_last_ckpt

        # group strategies
        if self.group_cfg is None or no_attr(self.group_cfg, "group_strategies"):
            self.group_func = None
        else:
            self.group_func = get_obj_from_str(self.group_cfg.group_strategies)


    def train(self, train_dataloader, train_use_list,
              val_dataloader=None, val_use_list=None,
              test_dataloader=None, test_use_list=None):
        
        run_data_dict = {}
        if self.log:
            self.log_and_print_train.create_general_file("train_general")
        if self.log and self.test_func_val is not None and val_dataloader is not None:
            self.log_and_print_val.create_general_file("val_general")
        if self.log and self.test_func_test is not None and test_dataloader is not None:
            self.log_and_print_test.create_general_file("test_general")

        if self.log and self.checkpoint_folder_path is not None:
            saved_model_file = OmegaConf.create()
            OmegaConf.save(saved_model_file, os.path.join(self.checkpoint_folder_path, 'saved_model.yaml'))

        if not self.log or self.checkpoint_folder_path is None:
            temp_save_file = os.path.join("temp", f"best_{time.time_ns()}.pth")


        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            self.log_and_print_train.create_detail_file(f'Epoch{epoch:04d}_details')
            header = f'Train Epoch[{epoch:04d}]: '
            self.log_and_print_train.init_model_run_logger(header, self.print_freq_train, self.print_summary_train)
            metric_logger = self.log_and_print_train.model_run_logger
            for _, v in self.add_meters_train.items():
                metric_logger.add_meter(**v)

            # train
            self.model.train()
            for batch_data in metric_logger.print_every(train_dataloader):
                run_data_dict = {v: batch_data[i] for i, v in enumerate(train_use_list)}
                outs = self.model_run_train(self.model, run_data_dict)
                if self.model_out_train is not None:
                    if isinstance(outs, tuple):
                        for i, k in enumerate(self.model_out_train):
                            run_data_dict[k] = outs[i]
                    else:
                        run_data_dict[self.model_out_train[0]] = outs

                loss_compute_input_dict = {k: run_data_dict[v] for k, v in self.loss_compute_args.items()}
                loss = self.loss_compute(**loss_compute_input_dict)
                
                if self.loss_compute_out is not None:
                    if isinstance(loss, tuple):
                        for i, k in enumerate(self.loss_compute_out):
                            run_data_dict[k] = loss[i]
                    else:
                        run_data_dict[self.loss_compute_out[0]] = loss

                for i,(_, meter_update_args) in enumerate(self.update_meters_train.items()):
                    meter_update_dict = {k: run_data_dict[v] if v in run_data_dict.keys() else v for k,v in meter_update_args.items()}
                    metric_logger.meters[self.add_meters_name_list_train[i]].update(**meter_update_dict)
                # print("batch end")
            self.log_and_print_train.log_model_run_info("train_general")

            # Val
            if self.test_func_val is not None and val_dataloader is not None:
                val_scores = self.test_func_val(self.model, self.save_path, self.log, model_weights_path=None,
                                                test_cfg=self.val_cfg, test_dataloader=val_dataloader, test_use_list=val_use_list,
                                                is_print=self.is_print,
                                                header=f'Val Epoch[{epoch:04d}]: ',
                                                post_process_header=f"Val Post Process Epoch[{epoch:04d}]: ",
                                                eval_header=f"Val Eval Epoch[{epoch:04d}]: ",
                                                general_file_name="val_general", detail_file_name=f'Epoch{epoch:04d}_details',
                                                device=self.device,
                                                log_and_print_test=self.log_and_print_val,
                                                print_freq=self.print_freq_val, print_summary=self.print_summary_val,
                                                log_summary=self.log_summary_val, print_post_process=self.print_post_process_val, 
                                                log_post_process=self.log_post_process_val,
                                                print_eval=self.print_eval_val, log_eval=self.log_eval_val,
                                                add_meters_name_list_test=self.add_meters_name_list_val,
                                                add_meters_test=self.add_meters_val,
                                                update_meters_test=self.update_meters_val,
                                                model_args_test=self.model_args_val,
                                                model_settings_test=self.model_settings_val,
                                                model_run_test=self.model_run_val,
                                                model_out_test=self.model_out_val,
                                                post_process_test=self.post_process_val,
                                                process_test=self.process_val,
                                                post_process_out_test=self.post_process_out_val,
                                                eval_func_test=self.eval_func_val,
                                                collect_data_test=self.collect_data_val,
                                                eval_summary_test=self.eval_summary_val,
                                                eval_out_test=self.eval_out_val)
                
            else:
                val_scores = None
            # print("val end")
            # Control stop
            if self.control_stop_func is not None and val_scores is not None:
                if self.control_stop_func.control(epoch, metric_logger, val_scores["results"]):
                    if self.log:
                        self.log_and_print_train.save_detail_file()
                    break

            # save model
            if self.log and self.checkpoint_folder_path is not None and not self.save_only_best_and_last_ckpt:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder_path, f"model_epoch{epoch:04d}.pth"))
            elif self.log and self.checkpoint_folder_path is not None and self.save_only_best_and_last_ckpt:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder_path, f"last.pth"))
                if self.control_stop_func is not None and self.control_stop_func.is_best():
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder_path, f"best.pth"))
            elif (not self.log or self.checkpoint_folder_path is None) and self.control_stop_func is not None and self.control_stop_func.is_best():
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                torch.save(self.model.state_dict(), temp_save_file)
            
            # update lastest saved-model epoch
            if self.log and self.checkpoint_folder_path is not None:
                saved_model_file["saved_epoch"] = epoch
                OmegaConf.save(saved_model_file, os.path.join(self.checkpoint_folder_path, 'saved_model.yaml'))

            # test
            if self.test_func_test is not None and test_dataloader is not None and not self.train_end_test:
                test_scores = self.test_func_val(self.model, self.save_path, self.log, model_weights_path=None,
                                                test_cfg=self.test_cfg, test_dataloader=test_dataloader, test_use_list=test_use_list,
                                                is_print=self.is_print,
                                                header=f'Test Epoch[{epoch:04d}]: ',
                                                post_process_header=f"Test Post Process Epoch[{epoch:04d}]: ",
                                                eval_header=f"Test Eval Epoch[{epoch:04d}]: ",
                                                general_file_name="test_general", detail_file_name=f'Epoch{epoch:04d}_details',
                                                device=self.device,
                                                log_and_print_test=self.log_and_print_test,
                                                print_freq=self.print_freq_test, print_summary=self.print_summary_test,
                                                log_summary=self.log_summary_test, print_post_process=self.print_post_process_test, 
                                                log_post_process=self.log_post_process_test,
                                                print_eval=self.print_eval_test, log_eval=self.log_eval_test,
                                                add_meters_name_list_test=self.add_meters_name_list_test,
                                                add_meters_test=self.add_meters_test,
                                                update_meters_test=self.update_meters_test,
                                                model_args_test=self.model_args_test,
                                                model_settings_test=self.model_settings_test,
                                                model_run_test=self.model_run_test,
                                                model_out_test=self.model_out_test,
                                                post_process_test=self.post_process_test,
                                                process_test=self.process_test,
                                                post_process_out_test=self.post_process_out_test,
                                                eval_func_test=self.eval_func_test,
                                                collect_data_test=self.collect_data_test,
                                                eval_summary_test=self.eval_summary_test,
                                                eval_out_test=self.eval_out_test)
                if self.control_stop_func is not None and self.control_stop_func.is_best():
                    best_test_scores = test_scores
            if self.log:
                self.log_and_print_train.save_detail_file()

            # optimizer scheduler lr
            if self.scheduler is not None:
                self.scheduler()
                # print(self.scheduler.scheduler.get_last_lr())
            # print("epoch end")

        if self.test_func_test is not None and \
            test_dataloader is not None and \
                self.train_end_test and \
                    self.control_stop_func is not None:
            if self.log and self.checkpoint_folder_path is not None:
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_folder_path, f"best.pth")))
            elif not self.log or self.checkpoint_folder_path is None:
                self.model.load_state_dict(torch.load(temp_save_file))
                os.remove(temp_save_file)

            test_scores = self.test_func_val(self.model, self.save_path, self.log, model_weights_path=None,
                                            test_cfg=self.test_cfg, test_dataloader=test_dataloader, test_use_list=test_use_list,
                                            is_print=self.is_print,
                                            header=f'Test Best Epoch (epoch {self.control_stop_func.get_best_epoch()}): ',
                                            post_process_header=f"Test Post Process Best Epoch (epoch {self.control_stop_func.get_best_epoch()}): ",
                                            eval_header=f"Test Eval Best Epoch (epoch {self.control_stop_func.get_best_epoch()}): ",
                                            general_file_name="test_general", detail_file_name=f'Best_details',
                                            device=self.device,
                                            log_and_print_test=self.log_and_print_test,
                                            print_freq=self.print_freq_test, print_summary=self.print_summary_test,
                                            log_summary=self.log_summary_test, print_post_process=self.print_post_process_test, 
                                            log_post_process=self.log_post_process_test,
                                            print_eval=self.print_eval_test, log_eval=self.log_eval_test,
                                            add_meters_name_list_test=self.add_meters_name_list_test,
                                            add_meters_test=self.add_meters_test,
                                            update_meters_test=self.update_meters_test,
                                            model_args_test=self.model_args_test,
                                            model_settings_test=self.model_settings_test,
                                            model_run_test=self.model_run_test,
                                            model_out_test=self.model_out_test,
                                            post_process_test=self.post_process_test,
                                            process_test=self.process_test,
                                            post_process_out_test=self.post_process_out_test,
                                            eval_func_test=self.eval_func_test,
                                            collect_data_test=self.collect_data_test,
                                            eval_summary_test=self.eval_summary_test,
                                            eval_out_test=self.eval_out_test)
            best_test_scores = test_scores
         
        if self.log:
                self.log_and_print_train.save_detail_file()

        if self.log:
            self.log_and_print_train.save_general_file()
        if self.log and self.test_func_val is not None and val_dataloader is not None:
            self.log_and_print_val.save_general_file()
        if self.log and self.test_func_test is not None and test_dataloader is not None:
            self.log_and_print_test.save_general_file()

        model_weights_path = None
        if self.control_stop_func is not None and self.test_func_val is not None and val_dataloader is not None:
            
            if not self.save_only_best_and_last_ckpt:
                model_weights_path = os.path.join(self.checkpoint_folder_path, 
                                                    f"model_epoch{self.control_stop_func.get_best_epoch():04d}.pth")

            else:
                model_weights_path = os.path.join(self.checkpoint_folder_path, 
                                                    f"best.pth")

            control_best = {}
            control_best["control_class"] = self.control_stop_func.get_control_class()
            control_best["best_epoch"] = self.control_stop_func.get_best_epoch()
            if not self.group:
                control_best.update(best_test_scores)
                control_best["results"]["control_score"] = self.control_stop_func.get_best_score()
            else:
                control_best["results"] = {}
        return control_best, model_weights_path