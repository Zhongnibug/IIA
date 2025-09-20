from work import work
from utils.get_attr import no_attr, convert_pos_to_dict_element, from_pos_add_element_to_dict, diff_default_conf
from utils.get_attr import from_pos_del_and_free
from utils.get_attr import compare_has_conf_is_equal
from utils.group_tools import merge_new_df, merge_temp_results
from omegaconf import OmegaConf
import pandas as pd
import os
import numpy as np
import pickle
import gc
import time
import warnings


def pass_and_work(configs, workdata, path_dir, group_cfg=None):
    # Ignore Warning
    warnings.filterwarnings('ignore')

    whether_branch = configs is not None and \
                    not no_attr(configs, "BRANCH") and \
                        not no_attr(configs["BRANCH"], "type") and \
                            not no_attr(configs["BRANCH"], "key") and \
                                not no_attr(configs["BRANCH"], "value")
    
    next_workdata = {}
    if whether_branch:
        group_path_list = ["BRANCH"] + configs["GROUPPATHLIST"][1:]
        group_name_list = [None] + configs["GROUPNAMELIST"][1:]
        next_configs = OmegaConf.create({
            "GROUP": True, 
            "group": configs["group"],
            "WORKFLOW": {
                "pass_and_work":{
                    "FUNCTION": configs["WORKFLOW"]["pass_and_work"]["FUNCTION"],
                    "INPUT": configs["WORKFLOW"]["pass_and_work"]["INPUT"],
                    "OUTPUT": [workdata["detail_word"][0]]
                }
            }
            })
    else:
        group_path_list = configs["GROUPPATHLIST"][1:]
        group_name_list = configs["GROUPNAMELIST"][1:]
        next_configs = OmegaConf.load(group_path_list[0])

    # whether add
    if no_attr(configs, "ADD"):
        configs["ADD"] = True

    # inherit
    if configs is None or no_attr(configs, "INHERIT"):
        group_inherit = OmegaConf.create({})
    else:
        group_inherit = configs["INHERIT"]
    
    # change inherit
    if next_configs is not None and not no_attr(next_configs, "INHERITCHANGE"):
        for k, v in next_configs["INHERITCHANGE"].items():
            x = convert_pos_to_dict_element({workdata["self_word"][0]: next_configs, workdata["workdata_word"][0]: next_workdata}, v, self_dic=next_configs)
            from_pos_add_element_to_dict({workdata["self_word"][0]: configs, workdata["workdata_word"][0]: workdata}, x, k, self_dic=configs)
    for k, v in group_inherit.items():
        x = convert_pos_to_dict_element({workdata["self_word"][0]: configs, workdata["workdata_word"][0]: workdata}, v, self_dic=configs)
        from_pos_add_element_to_dict({workdata["self_word"][0]: next_configs, workdata["workdata_word"][0]: next_workdata}, x, k, self_dic=next_configs)

    group_inherit = OmegaConf.create({k: k for k, _ in group_inherit.items()})
    if len(group_name_list) != 1 and no_attr(next_configs, "INHERIT"): 
        next_configs["INHERIT"] = group_inherit
    elif len(group_name_list) != 1 and not no_attr(next_configs, "INHERIT"):
        next_configs["INHERIT"] = OmegaConf.merge(group_inherit, next_configs["INHERIT"])

    if no_attr(next_configs, "BRANCHCONTINUE"):
        next_configs["BRANCHCONTINUE"] = configs["BRANCHCONTINUE"]
    else:
        next_configs["BRANCHCONTINUE"] = configs["BRANCHCONTINUE"] or next_configs["BRANCHCONTINUE"]

    # branch
    if whether_branch:
        
        configs["ADD"] = False
        next_configs["ADD"] = configs["ADD"]

        if configs["BRANCH"]["type"] == "range":
            if isinstance(configs["BRANCH"]["value"], int):
                branch_list = [i for i in range(configs["BRANCH"]["value"])]
            elif isinstance(configs["BRANCH"]["value"], list):
                if len(configs["BRANCH"]["value"]) == 1:
                    branch_list = [i for i in range(configs["BRANCH"]["value"][0])]
                elif len(configs["BRANCH"]["value"]) == 2:
                    branch_list = [i for i in range(configs["BRANCH"]["value"][0],
                                                    configs["BRANCH"]["value"][1])]
                elif len(configs["BRANCH"]["value"]) == 3:
                    branch_list = [i for i in range(configs["BRANCH"]["value"][0],
                                                    configs["BRANCH"]["value"][1],
                                                    configs["BRANCH"]["value"][2])]
                else:
                    raise Exception(f'The type {configs["BRANCH"]["type"]} does not support the list of length {len(configs["BRANCH"]["value"])}!!!')
            else:
                raise Exception(f'The type {configs["BRANCH"]["type"]} does not support the input of type {type(configs["BRANCH"]["value"])}!!!')
        elif configs["BRANCH"]["type"] == "list":
            branch_list = configs["BRANCH"]["value"]
        else:
            raise Exception(f'The type {configs["BRANCH"]["type"]} does not be supported!!!')
        
        branch_len = len(branch_list)
        tag_name = configs["BRANCH"]["key"].split('.')[-1]


        df = pd.DataFrame({})
        results_keys = []
        group_keys = []

        ver_path_dir = path_dir
        dir_tag = tag_name
        if os.path.exists(os.path.join(ver_path_dir, "branchinfo.yaml")):
            branchinfo = OmegaConf.load(os.path.join(ver_path_dir, "branchinfo.yaml"))
        else:
            branchinfo = OmegaConf.create({tag_name: {}})
            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))
        
        # if not configs["BRANCHCONTINUE"] or no_attr(branchinfo, tag_name):
        #     branchinfo[tag_name] = {}

        for i in range(branch_len):
            from_pos_add_element_to_dict({workdata["self_word"][0]: next_configs, workdata["workdata_word"][0]: next_workdata}, branch_list[i], pos=configs["BRANCH"]["key"], self_dic=next_configs)
            if len(group_name_list) != 1:
                if configs["BRANCH"]["key"] not in next_configs["INHERIT"].keys():
                    next_configs["INHERIT"][configs["BRANCH"]["key"]] = configs["BRANCH"]["key"]

            branchinfo = OmegaConf.load(os.path.join(ver_path_dir, "branchinfo.yaml"))
            if branch_list[i] not in branchinfo[tag_name].keys():
                workname = f"{tag_name}_{len(branchinfo[tag_name].keys())}"
            else:
                workname = branchinfo[tag_name][branch_list[i]]

            group_name_list[0] = workname

           
            work(next_configs, ver_path_dir, group_path_list, group_name_list, configs["GLOBAL"], deep=configs["DEEP"]+1,
                workname=workname, workdata=next_workdata, self_word=workdata["self_word"][0],
                workdata_word=workdata["workdata_word"][0],
                detail_word=workdata["detail_word"][0],
                update_keys=workdata["update_keys"],
                work_sleep=workdata["work_sleep"])
            
            if len(group_name_list) != 1:
                merge_temp_results(os.path.join(ver_path_dir, workname), workdata, False)
            
            branchinfo[tag_name][branch_list[i]] = workname

            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))
            
            dir_name = branch_list[i]
            
            new_results_keys = []
            new_group_keys = []
            
            if len(group_name_list) == 1:
                detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "workdetail.yaml"))
                if not no_attr(detail_cfg[workdata["detail_word"][0]], "results"):
                    new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results"].keys()) 
                    new_df = pd.DataFrame({k: [v] for k, v in \
                                            detail_cfg[workdata["detail_word"][0]]["results"].items()})
                else:
                    new_df = pd.DataFrame({})
             
            else:
                detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "temp_workdetail.yaml"))
                if not no_attr(detail_cfg[workdata["detail_word"][0]], "results_keys"):
                    new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results_keys"])
                if not no_attr(detail_cfg[workdata["detail_word"][0]], "group_keys"):
                    new_group_keys = list(detail_cfg[workdata["detail_word"][0]]["group_keys"])
                                      
                new_df = pd.read_csv(os.path.join(ver_path_dir, workname, "temp_results.csv"))


            if dir_tag not in new_group_keys:
                new_group_keys = [dir_tag] + new_group_keys

            new_df[dir_tag] = dir_name
            df, results_keys, group_keys = merge_new_df(df, new_df, results_keys, group_keys, new_results_keys, new_group_keys)

    elif not configs["ADD"]:

        next_configs["ADD"] = configs["ADD"]

        df = pd.DataFrame({})
        results_keys = []
        group_keys = []

        ver_path_dir = path_dir
        workname = group_name_list[0]

        dir_tag = f'dir_{configs["DEEP"]+1}'

        if os.path.exists(os.path.join(ver_path_dir, "branchinfo.yaml")):
            branchinfo = OmegaConf.load(os.path.join(ver_path_dir, "branchinfo.yaml"))
            if no_attr(branchinfo, dir_tag):
                branchinfo[dir_tag] = {}
                OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))
        else:
            branchinfo = OmegaConf.create({dir_tag: {}})
            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))

        if len(group_name_list) != 1 or \
            not configs["BRANCHCONTINUE"] or \
                workname not in branchinfo[dir_tag].keys():
            work(next_configs, ver_path_dir, group_path_list, group_name_list, configs["GLOBAL"], deep=configs["DEEP"]+1,
                workname=workname, workdata=next_workdata, self_word=workdata["self_word"][0],
                workdata_word=workdata["workdata_word"][0],
                detail_word=workdata["detail_word"][0],
                update_keys=workdata["update_keys"],
                work_sleep=workdata["work_sleep"])
            
            if len(group_name_list) != 1:
                merge_temp_results(os.path.join(ver_path_dir, workname), workdata, False)
            branchinfo[dir_tag][workname] = workname
            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))

        dir_name = workname
        
        new_results_keys = []
        new_group_keys = []
        
        if len(group_name_list) == 1:
            detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "workdetail.yaml"))
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "results"):
                new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results"].keys()) 
                new_df = pd.DataFrame({k: [v] for k, v in \
                                        detail_cfg[workdata["detail_word"][0]]["results"].items()})
            else:
                new_df = pd.DataFrame({})
            
        else:
            detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "temp_workdetail.yaml"))
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "results_keys"):
                new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results_keys"])
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "group_keys"):
                new_group_keys = list(detail_cfg[workdata["detail_word"][0]]["group_keys"])
                                    
            new_df = pd.read_csv(os.path.join(ver_path_dir, workname, "temp_results.csv"))


        if dir_tag not in new_group_keys:
            new_group_keys = [dir_tag] + new_group_keys

        new_df[dir_tag] = dir_name
        df, results_keys, group_keys = merge_new_df(df, new_df, results_keys, group_keys, new_results_keys, new_group_keys)


    else:
        if len(group_name_list) == 1:
            next_configs["ADD"] = False
        else:
            next_configs["ADD"] = configs["ADD"]


        df = pd.DataFrame({})
        results_keys = []
        group_keys = []

        current_version_idx = configs["VERSION"]
        ver_path_dir = os.path.join(path_dir, f"v{current_version_idx}")
        workname = group_name_list[0]

        dir_tag = f'dir_{configs["DEEP"]+1}'

        if os.path.exists(os.path.join(ver_path_dir, "branchinfo.yaml")):
            branchinfo = OmegaConf.load(os.path.join(ver_path_dir, "branchinfo.yaml"))
            if no_attr(branchinfo, dir_tag):
                branchinfo[dir_tag] = {}
                OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))
        else:
            branchinfo = OmegaConf.create({dir_tag: {}})
            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))

        if len(group_name_list) != 1 or \
            not configs["BRANCHCONTINUE"] or \
                workname not in branchinfo[dir_tag].keys():
            work(next_configs, ver_path_dir, group_path_list, group_name_list, configs["GLOBAL"], deep=configs["DEEP"]+1,
                workname=workname, workdata=next_workdata, self_word=workdata["self_word"][0],
                workdata_word=workdata["workdata_word"][0],
                detail_word=workdata["detail_word"][0],
                update_keys=workdata["update_keys"],
                work_sleep=workdata["work_sleep"])
            if len(group_name_list) != 1:
                merge_temp_results(os.path.join(ver_path_dir, workname), workdata, False)
            branchinfo[dir_tag][workname] = workname
            OmegaConf.save(branchinfo, os.path.join(ver_path_dir, "branchinfo.yaml"))

        dir_name = f"v{current_version_idx}.{workname}"
        
        
        new_results_keys = []
        new_group_keys = []
        
        if len(group_name_list) == 1:
            detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "workdetail.yaml"))
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "results"):
                new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results"].keys()) 
                new_df = pd.DataFrame({k: [v] for k, v in \
                                        detail_cfg[workdata["detail_word"][0]]["results"].items()})
            else:
                new_df = pd.DataFrame({})
            
        else:
            detail_cfg = OmegaConf.load(os.path.join(ver_path_dir, workname, "temp_workdetail.yaml"))
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "results_keys"):
                new_results_keys = list(detail_cfg[workdata["detail_word"][0]]["results_keys"])
            if not no_attr(detail_cfg[workdata["detail_word"][0]], "group_keys"):
                new_group_keys = list(detail_cfg[workdata["detail_word"][0]]["group_keys"])
                                    
            new_df = pd.read_csv(os.path.join(ver_path_dir, workname, "temp_results.csv"))


        if dir_tag not in new_group_keys:
            new_group_keys = [dir_tag] + new_group_keys

        new_df[dir_tag] = dir_name
        df, results_keys, group_keys = merge_new_df(df, new_df, results_keys, group_keys, new_results_keys, new_group_keys)

    df = df.set_index(group_keys)
    df.to_csv(os.path.join(path_dir, "temp_results.csv"))
    temp_workdetail_simple = OmegaConf.create({workdata["detail_word"][0]: {"results_keys": results_keys, "group_keys": group_keys}})
    OmegaConf.save(temp_workdetail_simple, os.path.join(path_dir, "temp_workdetail.yaml"))
    
    
    old_results_keys = []
    old_group_keys = []
    if os.path.exists(os.path.join(path_dir, "workdetail.yaml")):
        old_detail_cfg = OmegaConf.load(os.path.join(path_dir, "workdetail.yaml")) 
        if not no_attr(old_detail_cfg[workdata["detail_word"][0]], "results_keys"):
            old_results_keys = list(old_detail_cfg[workdata["detail_word"][0]]["results_keys"])
        if not no_attr(old_detail_cfg[workdata["detail_word"][0]], "group_keys"):
            old_group_keys = list(old_detail_cfg[workdata["detail_word"][0]]["group_keys"])
    
    temp_merge_results_keys = []
    temp_merge_group_keys = []
    for k in results_keys + old_results_keys:
        if k not in temp_merge_results_keys:
            temp_merge_results_keys.append(k)
    for k in group_keys + old_group_keys:
        if k not in temp_merge_group_keys:
            temp_merge_group_keys.append(k)

    return {"results_keys": temp_merge_results_keys, "group_keys": temp_merge_group_keys}
    
def mean_std_group_by_keys(path_dir, keys_dic, workdata, mean_key="seed", mean_value=100, saved_original=False):
    df = pd.read_csv(os.path.join(path_dir, "temp_results.csv"))
    temp_dcf = OmegaConf.load(os.path.join(path_dir, "temp_workdetail.yaml"))
    if not no_attr(temp_dcf[workdata["detail_word"][0]], "results_keys"):
        results_keys = list(temp_dcf[workdata["detail_word"][0]]["results_keys"])
    if not no_attr(temp_dcf[workdata["detail_word"][0]], "group_keys"):
        group_keys = list(temp_dcf[workdata["detail_word"][0]]["group_keys"]) 

    # if saved_original and os.path.exists(os.path.join(path_dir, "results_original.csv")):
    #     result_original_old = pd.read_csv(os.path.join(path_dir, "results_original.csv"))
    #     df_set_index = df.set_index([mean_key])
    #     mask = df_set_index.index.get_level_values(mean_key).notna()
    #     result_original_new = df_set_index[mask]
    #     result_original_new = result_original_new.dropna(axis=1, how='all')
    #     result_original_new = result_original_new.reset_index()
    #     result_original_concat = pd.concat([result_original_old, result_original_new], 
    #                                        join="outer", ignore_index=True)
    #     result_original_concat.to_csv(os.path.join(path_dir, "results_original.csv"))

    # elif saved_original and not os.path.exists(os.path.join(path_dir, "results_original.csv")):
    #     df.to_csv(os.path.join(path_dir, "results_original.csv"))

    if saved_original: 
        df.to_csv(os.path.join(path_dir, "temp_results_original.csv"))
    groupby_keys = []

    for k in group_keys:
        if k != mean_key:
            groupby_keys.append(k)
    df_multi_index = df.set_index(group_keys)
    if len(groupby_keys) > 0:
        df_mean = df_multi_index.groupby(groupby_keys).agg(["mean", "std"])
        df_mean.columns = ['.'.join(col) for col in df_mean.columns]
        df_mean = df_mean.reset_index()
        df_mean["mean_method"] = f"{mean_key}_{mean_value}"
    else:
        df_mean_v = df_multi_index.mean()
        df_std_v = df_multi_index.std()
        df_mean_dic = {}
        for key in list(df_mean_v.index):
            df_mean_dic['.'.join([key, "mean"])] = [df_mean_v[key]]
            df_mean_dic['.'.join([key, "std"])] = [df_std_v[key]]
        df_mean = pd.DataFrame(df_mean_dic)
        df_mean["mean_method"] = f"{mean_key}.{mean_value}"

    groupby_keys = ["mean_method"] + groupby_keys
    df_mean_set = df_mean.set_index(groupby_keys)
    
    df_mean_set.to_csv(os.path.join(path_dir, "temp_results.csv"))
    temp_dcf[workdata["detail_word"][0]]["results_keys"] = list(df_mean_set)
    temp_dcf[workdata["detail_word"][0]]["group_keys"] = groupby_keys
    OmegaConf.save(temp_dcf, os.path.join(path_dir, "temp_workdetail.yaml"))

    new_results_keys, new_group_keys = results_keys_and_group_keys_change(keys_dic["results_keys"], keys_dic["group_keys"],
                                                                          results_keys, group_keys,
                                                                          list(df_mean_set), groupby_keys)

    return {"results_keys": new_results_keys, "group_keys": new_group_keys}

def results_keys_and_group_keys_change(old_merge_results_keys, old_merge_group_keys,
                                       old_change_results_keys, old_change_group_keys,
                                       new_change_results_keys, new_change_group_keys):
    results_keys = []
    group_keys = []
    for k in old_merge_results_keys + new_change_results_keys:
        if k not in old_change_results_keys or k in new_change_results_keys:
            if k not in results_keys:
                results_keys.append(k)
    for k in old_merge_group_keys + new_change_group_keys:
        if k not in old_change_group_keys or k in new_change_group_keys:
            if k not in group_keys:
                group_keys.append(k)
    return results_keys, group_keys
    pass

def empty_opt():
    return {"results": {}}

def group_get_file_path(get_file_configs):
    if get_file_configs is None:
        raise Exception(f"The get_file_configs can not be Null!!!")
    if not no_attr(get_file_configs, "root_path"):
        root_path = get_file_configs.root_path
    else:
        raise Exception(f"The get_file_configs must contain root_path!!!")
    if not no_attr(get_file_configs, "compare_conf"):
        compare_conf = get_file_configs.compare_conf
    else:
        raise Exception(f"The get_file_configs must contain compare_conf!!!")
    if not no_attr(get_file_configs, "name"):
        name = get_file_configs.name
    else:
        raise Exception(f"The get_file_configs must contain name!!!")
    if not no_attr(get_file_configs, "suffix"):
        suffix = get_file_configs.suffix
    else:
        suffix = "pkl"
    if not no_attr(get_file_configs, "non_root_dir_list"):
        non_root_dir_list = get_file_configs.non_root_dir_list
    else:
        non_root_dir_list = []
    if not no_attr(get_file_configs, "sub_dir"):
        sub_dir = get_file_configs.sub_dir
    else:
        sub_dir = None
    if not no_attr(get_file_configs, "ban_dir_list"):
        ban_dir_list = get_file_configs.ban_dir_list
    else:
        ban_dir_list = []
    if not no_attr(get_file_configs, "strict_non_root_dir_list"):
        strict_non_root_dir_list = get_file_configs.strict_non_root_dir_list
    else:
        strict_non_root_dir_list = []

    file_whole_name = f"{name}.{suffix}"
    if sub_dir is not None:
        sub_dir_list = sub_dir.split('.')
        dir_path = sub_dir_list[0]
        for s_p in sub_dir_list[1:]:
            dir_path = os.path.join(dir_path, s_p)
        file_whole_name = os.path.join(dir_path, file_whole_name)
    file_path = get_file_path(root_path, compare_conf, file_whole_name, non_root_dir_list, strict_non_root_dir_list, ban_dir_list)

    if file_path is None:
        raise Exception(f"Do not exist the group file with config: {compare_conf}!!!")
    return file_path

def get_file_path(root_path, compare_conf, file_whole_name, non_root_dir_list=[], strict_non_root_dir_list=[], ban_dir_list=[]):

    file_path = None
    if os.path.exists(root_path) and os.path.exists(os.path.join(root_path, "version.yaml")):
        version_conf = OmegaConf.load(os.path.join(root_path, "version.yaml"))
        version_count = version_conf["count"]
        for i in range(version_count):
            if os.path.exists(os.path.join(root_path, f"v{i}")):
                ver_check_conf = OmegaConf.load(os.path.join(root_path, f"v{i}", "workdetail.yaml"))
                flag, strict_flag = compare_has_conf_is_equal(ver_check_conf, compare_conf)

                if strict_flag and os.path.exists(os.path.join(root_path, f"v{i}", file_whole_name)) and len(non_root_dir_list) == 0 and len(strict_non_root_dir_list) == 0:
                    return os.path.join(root_path, f"v{i}", file_whole_name)
                elif flag:
                    if len(strict_non_root_dir_list) != 0:
                        if os.path.exists(os.path.join(root_path, f"v{i}", strict_non_root_dir_list[0])):
                            file_path = get_file_path(os.path.join(root_path, f"v{i}", strict_non_root_dir_list[0]),
                                                    compare_conf,
                                                    file_whole_name,
                                                    non_root_dir_list=non_root_dir_list,
                                                    strict_non_root_dir_list=strict_non_root_dir_list[1:],
                                                    ban_dir_list=ban_dir_list)
                        return file_path
                    elif len(non_root_dir_list) == 0:
                        with os.scandir(os.path.join(root_path, f"v{i}")) as entries:
                            for entry in entries:
                                if entry.is_dir() and entry.name not in ban_dir_list:
                                    file_path = get_file_path(os.path.join(root_path, f"v{i}", entry.name),
                                                            compare_conf,
                                                            file_whole_name,
                                                            non_root_dir_list=[],
                                                            strict_non_root_dir_list=[],
                                                            ban_dir_list=ban_dir_list)
                                    if file_path is not None:
                                        return file_path
                    else:
                        with os.scandir(os.path.join(root_path, f"v{i}")) as entries:
                            entries_dir_name = []
                            for entry in entries:
                                if entry.is_dir():
                                    entries_dir_name.append(entry.name)
                            if non_root_dir_list[0] in entries_dir_name:
                                file_path = get_file_path(os.path.join(root_path, f"v{i}", non_root_dir_list[0]),
                                                        compare_conf,
                                                        file_whole_name,
                                                        non_root_dir_list=non_root_dir_list[1:],
                                                        strict_non_root_dir_list=[],
                                                        ban_dir_list=ban_dir_list)
                                if file_path is not None:
                                    return file_path
                            for entry_name in entries_dir_name:
                                if entry_name != non_root_dir_list[0] and entry_name not in ban_dir_list:
                                    file_path = get_file_path(os.path.join(root_path, f"v{i}", entry_name),
                                                            compare_conf,
                                                            file_whole_name,
                                                            non_root_dir_list=non_root_dir_list,
                                                            strict_non_root_dir_list=[],
                                                            ban_dir_list=ban_dir_list)
                                    if file_path is not None:
                                        return file_path
    elif os.path.exists(root_path):
        flag = True
        strict_flag = False
        if os.path.exists(os.path.join(root_path, "original.yaml")):
            ver_check_conf = OmegaConf.load(os.path.join(root_path, "original.yaml"))
            flag, strict_flag = compare_has_conf_is_equal(ver_check_conf, compare_conf)

        if strict_flag and os.path.exists(os.path.join(root_path, file_whole_name)) and len(non_root_dir_list) == 0 and len(strict_non_root_dir_list) == 0:
            return os.path.join(root_path, file_whole_name)
        elif flag:
            if len(strict_non_root_dir_list) != 0:
                if os.path.exists(os.path.join(root_path, strict_non_root_dir_list[0])):
                    file_path = get_file_path(os.path.join(root_path, strict_non_root_dir_list[0]),
                                                compare_conf,
                                                file_whole_name,
                                                non_root_dir_list=non_root_dir_list,
                                                strict_non_root_dir_list=strict_non_root_dir_list[1:],
                                                ban_dir_list=ban_dir_list)
                return file_path
            elif len(non_root_dir_list) == 0:
                with os.scandir(root_path) as entries:
                    for entry in entries:
                        if entry.is_dir() and entry.name not in ban_dir_list:
                            file_path = get_file_path(os.path.join(root_path, entry.name),
                                                        compare_conf,
                                                        file_whole_name,
                                                        non_root_dir_list=[],
                                                        strict_non_root_dir_list=[],
                                                        ban_dir_list=ban_dir_list)
                            if file_path is not None:
                                return file_path
            else:
                with os.scandir(root_path) as entries:
                    entries_dir_name = []
                    for entry in entries:
                        if entry.is_dir():
                            entries_dir_name.append(entry.name)
                    if non_root_dir_list[0] in entries_dir_name:
                        file_path = get_file_path(os.path.join(root_path, non_root_dir_list[0]),
                                                  compare_conf,
                                                  file_whole_name,
                                                  non_root_dir_list=non_root_dir_list[1:],
                                                  strict_non_root_dir_list=[],
                                                  ban_dir_list=ban_dir_list)
                        if file_path is not None:
                            return file_path

                    for entry_name in entries_dir_name:
                        if entry_name != non_root_dir_list[0] and entry_name not in ban_dir_list:
                            file_path = get_file_path(os.path.join(root_path, entry_name),
                                                        compare_conf,
                                                        file_whole_name,
                                                        non_root_dir_list=non_root_dir_list,
                                                        strict_non_root_dir_list=[],
                                                        ban_dir_list=ban_dir_list)
                            if file_path is not None:
                                return file_path
    return file_path

