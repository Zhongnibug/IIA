import omegaconf
from omegaconf import OmegaConf
import torch
import random
import numpy as np
import os
from datetime import datetime
import psutil
import operator

from utils.get_attr import no_attr
from utils.get_attr import convert_pos_to_dict_element
from utils.get_attr import from_pos_add_element_to_dict
from utils.get_attr import add_omgaconf_from_other
from utils.get_attr import diff_default_conf
from utils.get_obj import get_obj_from_str
from utils.group_tools import merge_temp_results

from subwork import subwork


def work(configs: omegaconf.dictconfig.DictConfig, 
         path_dir,
         paths_list,
         names_list,
         global_configs,
         deep=0,
         workname = None,
         workdata = {},
         self_word="SELF",
         workdata_word="WORKDATA",
         detail_word="DETAIL",
         update_keys=["WORKDETAIL"],
         work_sleep=None):

    # Copy original configs
    # print(type(configs))
    if isinstance(configs, omegaconf.dictconfig.DictConfig):
        original_configs = configs
    else:
        original_configs = OmegaConf.create(configs)

    # log or print
    configs["GLOBAL"] = global_configs
    configs["LOG"] = global_configs["LOG"]
    configs["PRINT"] = global_configs["PRINT"]
    configs["NONCHECKKEYS"] = global_configs["NONCHECKKEYS"]
    configs["PARENTPATH"] = path_dir
    configs["GROUPPATHLIST"] = paths_list
    configs["GROUPNAMELIST"] = names_list
    configs["DEEP"] = deep
    if workname is None:
        configs["WORKNAME"] = configs["GROUPNAMELIST"][0]
    else:
        configs["WORKNAME"] = workname

    workdata["self_word"] = [self_word]
    workdata["workdata_word"] = [workdata_word]
    workdata["detail_word"] = [detail_word]
    workdata["update_keys"] = update_keys
    workdata["work_sleep"] = work_sleep

    if workdata["work_sleep"] is not None:
        workdata["work_sleep"].sleep()
    # Get final work name
    # if workname is None:
    #     current_time = datetime.now()
    #     formatted_date_time = current_time.strftime('%Y%m%d%H%M%S')
    #     final_work_name = f'{formatted_date_time}_{configs.WORKNAME}'
    # else:
    #     final_work_name = configs.WORKNAME
    final_work_name = configs.WORKNAME
    # Get save folder

    configs["MAINPATH"] = os.path.join(path_dir, final_work_name)
    if configs.LOG and not os.path.exists(configs["MAINPATH"]):
        os.makedirs(configs["MAINPATH"])

    version_conf = None
    flag_version_detail = False
    if configs.LOG and not no_attr(configs, "GROUP") and configs["GROUP"]:
        if (no_attr(configs, "BRANCH") or \
            no_attr(configs["BRANCH"], "type") or \
                no_attr(configs["BRANCH"], "key") or \
                    no_attr(configs["BRANCH"], "value")) and \
                        (no_attr(configs, "ADD") or configs["ADD"]):
            
            path_dir = configs["MAINPATH"]
            
            if os.path.exists(os.path.join(path_dir, "version.yaml")):
                version_conf = OmegaConf.load(os.path.join(path_dir, "version.yaml"))
                default_conf = version_conf["v0"]

            else:
                version_conf = OmegaConf.create({"count": 0})
                default_conf = original_configs

            new_version_idx = version_conf["count"]

            current_version_idx = -1
            non_check_keys_conf = OmegaConf.create({n_c_k: None for n_c_k in configs["NONCHECKKEYS"]})
            default_check_conf = OmegaConf.merge(default_conf, non_check_keys_conf)
            check_conf = OmegaConf.merge(configs, non_check_keys_conf)
            for v_i in range(new_version_idx):
                if v_i == 0:
                    if default_check_conf == check_conf:
                        current_version_idx = 0
                        break
                else:
                    ver_check_conf = OmegaConf.merge(default_check_conf, version_conf[f"v{v_i}"])
                    if ver_check_conf == check_conf:
                        current_version_idx = v_i
                        break

            if current_version_idx == -1 and new_version_idx >0:
                diffs = diff_default_conf(default_check_conf, check_conf)
                diffs_conf = {}
                for diff in diffs:
                    x = check_conf
                    find_diffs_conf = diffs_conf
                    for diff_k in diff[:-1]:
                        if no_attr(find_diffs_conf, diff_k):
                            find_diffs_conf[diff_k] = {}
                        find_diffs_conf = find_diffs_conf[diff_k]
                        x = x[diff_k]
                    if no_attr(x, diff[-1]):
                        find_diffs_conf[diff[-1]] = None
                    else:
                        find_diffs_conf[diff[-1]] = x[diff[-1]]
                flag_version_detail = True
                current_version_idx = new_version_idx
                version_conf["count"] = new_version_idx + 1
                version_conf[f"v{current_version_idx}"] = diffs_conf
            elif current_version_idx == -1 and new_version_idx ==0:
                flag_version_detail = True
                current_version_idx = new_version_idx
                version_conf["count"] = new_version_idx + 1
                version_conf["v0"] = default_conf
            configs["VERSION"] = current_version_idx
            configs["WORKPATH"] = os.path.join(path_dir, f"v{current_version_idx}")          
        else:
            configs["WORKPATH"] = configs["MAINPATH"]
    else:
        configs["WORKPATH"] = configs["MAINPATH"]

    if configs.LOG and not os.path.exists(configs["WORKPATH"]):
        os.makedirs(configs["WORKPATH"])

    current_time = datetime.now()
    formatted_date_time = current_time.strftime('%Y.%m.%d %H:%M:%S')
    begin_log = f"[{formatted_date_time}]: Begin to {final_work_name} !!!"
    print(begin_log)

    # Begin to work

    work_flow = {}

    if configs.LOG:
        work_detail = {}
    else:
        work_detail = None

    for subwork_key, subwork_value in configs.WORKFLOW.items():
        
        func_name = subwork(dic={self_word: configs, workdata_word: workdata},
                            subwork_value=subwork_value,
                            subwork_key=subwork_key,
                            self_dic=None,
                            work_detail=work_detail,
                            detail_word=detail_word)
        
        input_keys = "None" if no_attr(subwork_value, "INPUT") else ",".join(list(subwork_value["INPUT"].keys()))
        output_keys = "None" if no_attr(subwork_value, "OUTPUT") else ",".join(list(subwork_value["OUTPUT"]))

        work_flow[func_name] = f"{input_keys} ==> {output_keys}"

        if configs.PRINT:
            print(f'{func_name}: {work_flow[func_name]}')

    # Save WORKDETAIL

    if configs.LOG:
        workflow_info = OmegaConf.create(work_flow)
        OmegaConf.save(workflow_info, os.path.join(configs["MAINPATH"], "workflow.yaml"))

        workdetail_configs = OmegaConf.create({
            workdata["detail_word"][0]: work_detail
            })
        add_omgaconf_from_other(workdetail_configs, configs, workdata["update_keys"])
        OmegaConf.save(workdetail_configs, os.path.join(configs["MAINPATH"], "workdetail.yaml"))
        
        if flag_version_detail:
            OmegaConf.save(workdetail_configs, os.path.join(configs["WORKPATH"], "workdetail.yaml"))

        if version_conf is not None:
            OmegaConf.save(version_conf, os.path.join(path_dir, "version.yaml"))  
        if  not os.path.exists(os.path.join(configs["MAINPATH"], f'{configs["GROUPNAMELIST"][0]}.yaml')) or \
            (not no_attr(configs, "ADD") and not configs["ADD"]):
            OmegaConf.save(original_configs, os.path.join(configs["MAINPATH"], "original.yaml"))

        if configs["DEEP"] == 0:
            merge_temp_results(configs["MAINPATH"], workdata)
    
    current_time = datetime.now()
    formatted_date_time = current_time.strftime('%Y.%m.%d %H:%M:%S')
    end_log = f"[{formatted_date_time}]: End to {final_work_name} !!!"
    print(end_log)

