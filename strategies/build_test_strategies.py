from strategies.transrormer_mask_strategies import subsequent_mask_2d
import torch
import psutil
from utils.get_attr import no_attr
from utils.get_attr import convert_all_f64_to_f32
from utils.get_obj import get_obj_from_str


def auto_test(model, save_path, log, model_weights_path=None,
              test_cfg=None, test_dataloader=None, test_use_list=None,
              is_print=True,
              header="Test:", post_process_header="Post Process Infos:", eval_header="Eval:",
              general_file_name="test_general", detail_file_name="test_detail",
              device=None, log_and_print_test=None, print_freq=None, print_summary=None,
              log_summary=None, print_post_process=None, log_post_process=None,
              print_eval=None, log_eval=None,
              add_meters_name_list_test=None, add_meters_test=None, update_meters_test=None,
              model_args_test=None, model_settings_test=None, model_run_test=None, model_out_test=None,
              post_process_test=None, process_test=None, post_process_out_test=None,
              eval_func_test=None, collect_data_test=None, eval_summary_test=None, eval_out_test=None):
    if device is None and no_attr(test_cfg, "device"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device is None and not no_attr(test_cfg, "device"):
        device = test_cfg.device

    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path))
        model.to(device)
    else:
        model.to(device)

    if log_and_print_test is None and no_attr(test_cfg, "log_print_strategies"):
        raise Exception("Do not define log_print_strategies in the configs of test!!!")
    elif log_and_print_test is None:
        if no_attr(test_cfg, "log_print_delimiter"):
            log_and_print_test = get_obj_from_str(test_cfg.log_print_strategies)(save_path, log=log)
        else:
            log_and_print_test = get_obj_from_str(test_cfg.log_print_strategies)(save_path, delimiter=test_cfg.log_print_delimiter, log=log)

    if not is_print:
        print_freq = None
    elif print_freq is None and not no_attr(test_cfg.print, "print_freq"):
        print_freq = test_cfg.print.print_freq
    
    if not is_print:
       print_summary = False 
    elif print_summary is None and (no_attr(test_cfg, "print") or no_attr(test_cfg.print, "print_summary")):
        print_summary = True
    elif print_summary is None:
        print_summary = test_cfg.print.print_summary

    if log_summary is None and (no_attr(test_cfg, "log") or no_attr(test_cfg.log, "log_summary")):
        log_summary = True
    elif log_summary is None:
        log_summary = test_cfg.log.log_summary

    if not is_print:
        print_post_process = False
    elif print_post_process is None and (no_attr(test_cfg, "print") or no_attr(test_cfg.print, "print_post_process")):
        print_post_process = False
    elif print_post_process is None:
        print_post_process = test_cfg.print.print_post_process

    if log_post_process is None and (no_attr(test_cfg, "log") or no_attr(test_cfg.log, "log_post_process")):
        log_post_process = False
    elif log_post_process is None:
        log_post_process = test_cfg.log.log_post_process

    if not is_print:
        print_eval = False
    elif print_eval is None and (no_attr(test_cfg, "print") or no_attr(test_cfg.print, "print_eval")):
        print_eval = False
    elif print_eval is None:
        print_eval = test_cfg.print.print_eval

    if log_eval is None and (no_attr(test_cfg, "log") or no_attr(test_cfg.log, "log_eval")):
        log_eval = False
    elif log_eval is None:
        log_eval = test_cfg.log.log_eval

    if add_meters_test is None and no_attr(test_cfg, "meters"):
        add_meters_test = {}
        add_meters_name_list_test = []
    elif add_meters_test is None:
        add_meters_test = {k: v["add"] for k, v in test_cfg.meters.items()}
        add_meters_name_list_test = [v["name"] for _,v in add_meters_test.items()]

    if update_meters_test is None and no_attr(test_cfg, "meters"):
        update_meters_test = {}
    elif update_meters_test is None:
        update_meters_test = {k: v["update"] for k, v in test_cfg.meters.items()} 

    log_and_print_test.init_model_run_logger(header, print_freq, print_summary)
    metric_logger = log_and_print_test.model_run_logger
    for _, v in add_meters_test.items():
        metric_logger.add_meter(**v)

    if model_args_test is None and \
        (no_attr(test_cfg, "model_run") or \
         no_attr(test_cfg.model_run, "model_args")):
        raise Exception(f"{header} The model_args of Test is not defined!!!")
    elif model_args_test is None:
        model_args_test = test_cfg.model_run.model_args

    if model_settings_test is None and \
        (no_attr(test_cfg, "model_run") or \
         no_attr(test_cfg.model_run, "model_settings")):
        model_settings_test = {}
    elif model_settings_test is None:
        model_settings_test = test_cfg.model_run.model_settings

    if model_run_test is None and no_attr(test_cfg, "model_run"):
        raise Exception(f"{header} The model_run func of Test is not defined!!!")
    elif model_run_test is None:
            model_run_strategy_test = test_cfg.model_run.model_run_strategies
            model_run_test = getattr(get_obj_from_str(model_run_strategy_test)(
                    muti_func_name_dict = test_cfg.model_run.model_run_func,
                    muti_settings_dict = model_settings_test,
                    muti_device_dict = device,
                    muti_model_args = model_args_test
            ), "model_run")

    if model_out_test is None and \
        (no_attr(test_cfg, "model_run") or \
         no_attr(test_cfg.model_run, "model_out")):
        model_out_test = None
    elif model_out_test is None:
        model_out_test = test_cfg.model_run.model_out

    if post_process_test is None and \
        (no_attr(test_cfg, "post_process") or \
         no_attr(test_cfg.post_process, "post_process_strategies")):
        post_process_test = None
    elif post_process_test is None and no_attr(test_cfg.post_process, "init_args"):
        post_process_test = get_obj_from_str(test_cfg.post_process.post_process_strategies)()
    elif post_process_test is None:
        post_process_test = get_obj_from_str(test_cfg.post_process.post_process_strategies)(**test_cfg.post_process.init_args)

    if process_test is None and \
        (no_attr(test_cfg, "post_process") or \
         no_attr(test_cfg.post_process, "process")):
        process_test = {}
    elif process_test is None:
        process_test = test_cfg.post_process.process

    if post_process_out_test is None and \
        (no_attr(test_cfg, "post_process") or \
         no_attr(test_cfg.post_process, "post_process_out")):
        post_process_out_test = []
    elif post_process_out_test is None:
        post_process_out_test = test_cfg.post_process.post_process_out

    log_and_print_test.init_post_process_logger(post_process_header, print_post_process)
    post_process_logger = log_and_print_test.post_process_logger

    if eval_func_test is None and \
        (no_attr(test_cfg, "eval") or \
         no_attr(test_cfg.eval, "build_eval_strategies")):
        eval_func_test = None
    elif eval_func_test is None and no_attr(test_cfg.eval, "init_args"):
        eval_func_test = get_obj_from_str(test_cfg.eval.build_eval_strategies)
    elif eval_func_test is None:
        eval_func_test = get_obj_from_str(test_cfg.eval.build_eval_strategies)(**test_cfg.eval.init_args)

    if eval_func_test is not None:
        eval_func_test.clear()

    if collect_data_test is None and \
        (no_attr(test_cfg, "eval") or \
         no_attr(test_cfg.eval, "collect_data")):
        collect_data_test = {}
    elif collect_data_test is None:
        collect_data_test = test_cfg.eval.collect_data

    if eval_summary_test is None and \
        (no_attr(test_cfg, "eval") or \
         no_attr(test_cfg.eval, "eval_summary")):
        eval_summary_test = {}
    elif eval_summary_test is None:
        eval_summary_test = test_cfg.eval.eval_summary

    if eval_out_test is None and \
        (no_attr(test_cfg, "eval") or \
         no_attr(test_cfg.eval, "eval_out")):
        eval_out_test = []
    elif eval_out_test is None:
        eval_out_test = test_cfg.eval.eval_out

    log_and_print_test.init_eval_logger(eval_header, print_eval)
    eval_logger = log_and_print_test.eval_logger

    if log:
        whether_close_general_file = log_and_print_test.create_general_file(general_file_name)
        whether_close_detail_file = log_and_print_test.create_detail_file(detail_file_name)
    else:
        whether_close_general_file = False
        whether_close_detail_file = False     

    # test
    model.eval()
    run_data_dict = {}
    for batch_data in metric_logger.print_every(test_dataloader):
        run_data_dict = {v: batch_data[i] for i, v in enumerate(test_use_list)}
        
        outs = model_run_test(model, run_data_dict)
        if model_out_test is not None:
            if isinstance(outs, tuple):
                for i, k in enumerate(model_out_test):
                    run_data_dict[k] = outs[i]
            else:
                run_data_dict[model_out_test[0]] = outs

        process_args = {k: run_data_dict[v] for k,v in process_test.items()}
        if post_process_test is not None:
            post_process_res = post_process_test.process(post_process_logger, **process_args)
        
        if post_process_out_test is not None:
            if isinstance(post_process_res, tuple):
                for i, k in enumerate(post_process_out_test):
                    run_data_dict[k] = post_process_res[i]
            else:
                run_data_dict[post_process_out_test[0]] = post_process_res

        collect_args = {k: run_data_dict[v] for k,v in collect_data_test.items()}
        if eval_func_test is not None:
            eval_func_test.collect_data(**collect_args)

        for i,(_, meter_update_args) in enumerate(update_meters_test.items()):
            meter_update_dict = {k: run_data_dict[v] if v in run_data_dict.keys() else v for k,v in meter_update_args.items()}
            metric_logger.meters[add_meters_name_list_test[i]].update(**meter_update_dict)
    
    # log summary
    if log and log_summary:
        log_and_print_test.log_model_run_info(general_file_name)

    # eval summary
    eval_summary_args = {k: run_data_dict[v] for k,v in eval_summary_test.items()}
    scores = eval_func_test.eval_summary(eval_logger, **eval_summary_args)

    if isinstance(scores, tuple):
        for i, k in enumerate(eval_out_test):
            run_data_dict[k] = scores[i]
    else:
        run_data_dict[eval_out_test[0]] = scores
    
    # log eval
    if log and log_eval:
        log_and_print_test.log_eval_info(general_file_name)
    
    # log post process
    if log and log_post_process:
        log_and_print_test.log_post_process_info()

    # save file
    if whether_close_general_file:
        log_and_print_test.save_general_file()
    if whether_close_detail_file:
        log_and_print_test.save_detail_file()
    
    return {"results": convert_all_f64_to_f32(scores)}