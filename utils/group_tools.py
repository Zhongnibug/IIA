from omegaconf import OmegaConf
import pandas as pd
import os
import numpy as np

from utils.get_attr import no_attr

def merge_temp_results(merge_main_path, workdata, is_delete_temp=False):
    
    path_dir = merge_main_path

    results_keys = []
    group_keys = []

    temp_results_keys = []
    temp_group_keys = []
    
    # load df
    if os.path.exists(os.path.join(path_dir, "workdetail.yaml")):
        existed_detail_conf = OmegaConf.load(os.path.join(path_dir, "workdetail.yaml"))
        if not no_attr(existed_detail_conf[workdata["detail_word"][0]], "results_keys"):
            results_keys = list(existed_detail_conf[workdata["detail_word"][0]]["results_keys"])
        if not no_attr(existed_detail_conf[workdata["detail_word"][0]], "group_keys"):            
            group_keys = list(existed_detail_conf[workdata["detail_word"][0]]["group_keys"])
        if os.path.exists(os.path.join(path_dir, "results.csv")):     
            df = pd.read_csv(os.path.join(path_dir, "results.csv"))
        else:
            df = pd.DataFrame({})
    else:
        existed_detail_conf = OmegaConf.create({workdata["detail_word"][0]:
                                                {"results_keys":[], "group_keys":[]}})
        
        df = pd.DataFrame({})

    # load new df
    if os.path.exists(os.path.join(path_dir, "temp_workdetail.yaml")):
        existed_temp_detail_conf = OmegaConf.load(os.path.join(path_dir, "temp_workdetail.yaml"))
        if not no_attr(existed_temp_detail_conf[workdata["detail_word"][0]], "results_keys"):
            temp_results_keys = list(existed_temp_detail_conf[workdata["detail_word"][0]]["results_keys"])
        if not no_attr(existed_temp_detail_conf[workdata["detail_word"][0]], "group_keys"):            
            temp_group_keys = list(existed_temp_detail_conf[workdata["detail_word"][0]]["group_keys"])
        if os.path.exists(os.path.join(path_dir, "temp_results.csv")):     
            new_df = pd.read_csv(os.path.join(path_dir, "temp_results.csv"))
        else:
            new_df = pd.DataFrame({})
    else:
        new_df = pd.DataFrame({})

    df, merge_results_keys, merge_group_keys = merge_new_df(df, new_df, results_keys, group_keys, temp_results_keys, temp_group_keys)
    df = df.set_index(merge_group_keys)
    df.to_csv(os.path.join(path_dir, "results.csv"))

    existed_detail_conf[workdata["detail_word"][0]]["results_keys"] = merge_results_keys
    existed_detail_conf[workdata["detail_word"][0]]["group_keys"] = merge_group_keys
    OmegaConf.save(existed_detail_conf, os.path.join(path_dir, "workdetail.yaml"))

    if is_delete_temp:
        os.remove(os.path.join(path_dir, "temp_workdetail.yaml"))
        os.remove(os.path.join(path_dir, "temp_results.csv"))
    pass      

def merge_new_df(df, new_df, results_keys, group_keys, new_results_keys, new_group_keys):
    merge_results_keys = []
    for k in results_keys + new_results_keys:
        if k not in merge_results_keys:
            merge_results_keys.append(k)

    merge_group_keys = []
    for k in group_keys + new_group_keys:
        if k not in merge_group_keys:
            merge_group_keys.append(k)

    df_loss_group_keys = []
    for k in new_group_keys:
        if k not in list(df.columns):
            df_loss_group_keys.append(k)
    
    new_df_loss_group_keys = []
    for k in group_keys:
        if k not in list(new_df.columns):
            new_df_loss_group_keys.append(k)

    if df.empty:
        df = new_df
        for k in new_df_loss_group_keys:
            df[k] = np.nan
    elif not new_df.empty:
        for k in df_loss_group_keys:
            df[k] = np.nan
        
        df = df.fillna("NaN")
        df = df.set_index(merge_group_keys)
        df = df.sort_index()
        
        for k in new_df_loss_group_keys:
            new_df[k] = np.nan

        new_df = new_df.fillna("NaN")
        new_df = new_df.set_index(merge_group_keys)
        new_df = new_df.sort_index()

        for index in new_df.index:
            for k in new_results_keys:
                df.loc[index, k] = new_df.loc[index, k]

        df = df.reset_index()
        df = df.where(df != "NaN", np.nan)
    else:
        pass

    return df, merge_results_keys, merge_group_keys