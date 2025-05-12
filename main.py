from omegaconf import OmegaConf
import argparse
from work import work
import os
from utils.get_attr import no_attr

import time
from datetime import datetime

SLEEP_INTERVAL = 3600
SLEEP_TIME = 600

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', '-c', type=lambda x: x.split(','), default="",
                        help="The yaml file paths of group and work which seperated by ','.")
    parser.add_argument('--workname', '-w', type=lambda x: x.split(','), default="")

    parser.add_argument('--globalname', '-g',type=lambda x: x.split(','), default="base.yaml",
                        help="The yaml files in configs/global of global configs which seperated by ','.")
    parser.add_argument('--global_path', type=lambda x: x.split(','), default="",
                        help="The yaml file paths of global configs which seperated by ','.")
    parser.add_argument("--branch_continue", '-b', action='store_true',
                        help="If used, the branches will be generated if they didn't exist. \
                            Otherwise, all branches will be regenerated.")

    args = parser.parse_args()
    # args.branch_continue = True
    return args

def find_path_and_name(path, name, default_dir="configs"):
        if path == '' and name == '':
            return False, None, None
        elif path == '' and name.rsplit(".", 1)[-1] == "yaml":
            new_path = os.path.join(default_dir, name)
        elif path == '' and name.rsplit(".", 1)[-1] != "yaml":
            new_path = os.path.join(default_dir, f"{name}.yaml")

        if name == "":
            new_name = path.split("/")[-1].rsplit(".", 1)[0]
        elif name.rsplit(".", 1)[-1] == "yaml":
            new_name = name.rsplit(".", 1)[0]
        else:
            new_name = name
        return True, new_path, new_name

def get_path_name_lists(args):
    # group
    group_len = max(len(args.cfg_path), len(args.workname))
    if len(args.cfg_path) < group_len:
        args.cfg_path += [''] * (group_len - len(args.cfg_path))
    if len(args.workname) < group_len:
        args.workname += [''] * (group_len - len(args.workname))
    
    for i in range(group_len):
        flag, args.cfg_path[i], args.workname[i] = find_path_and_name(
            args.cfg_path[i], args.workname[i], default_dir="configs")
        if not flag:
            raise Exception(f"The {i}-th input of --cfg_path and --workname are Null!!!")
        
    # global
    global_len = max(len(args.global_path), len(args.globalname))
    if len(args.global_path) < global_len:
        args.global_path += [''] * (group_len - len(args.global_path))
    if len(args.globalname) < global_len:
        args.globalname += [''] * (group_len - len(args.globalname))

    temp_global_paths = []
    temp_globalnames = []
    for i in range(global_len):

        flag, temp_global_path, temp_globalname = find_path_and_name(
            args.global_path[i], args.globalname[i], default_dir="configs/global")
        if flag:
            temp_global_paths.append(temp_global_path)
            temp_globalnames.append(temp_globalname)
    args.global_path = temp_global_paths
    args.globalname = temp_globalnames    

    return args

def group_check(args):

    # work check
    work_configs = OmegaConf.load(args.cfg_path[-1])        
    if not no_attr(work_configs, "GROUP") and work_configs.GROUP:
        return False, -1

    # group check
    for i, p in enumerate(args.cfg_path[:-1]):
        group_cfgs = OmegaConf.load(p)
        if no_attr(group_cfgs, "GROUP") or not group_cfgs.GROUP:
            return False, i

    return True, -2

class WorkSleep:
    def __init__(self, sleep_interval=60*60, sleep_time=60*10):
        super(WorkSleep, self).__init__()
        self.sleep_interval = sleep_interval
        self.sleep_time = sleep_time
        self.last_time = datetime.now()

    def sleep(self):
        now_time = datetime.now()
        time_difference = now_time - self.last_time
        if time_difference.total_seconds() >= self.sleep_interval:
            print(f"Sleeping for {self.sleep_time} seconds...")
            time.sleep(self.sleep_time)
            print("Awake now.")
            self.last_time = datetime.now()

if __name__ == "__main__":

    # load args
    args = load_args()

    # get cfg paths and work names
    args = get_path_name_lists(args)

    # check group
    flag, pos = group_check(args)
    if not flag and pos == -1:
        raise Exception("The final config must be work config rather than group config!!!")
    elif not flag:
        raise Exception(f"The {pos}-th config must be group config rather than work config!!!")
    
    # load global configs
    global_configs = OmegaConf.create()
    for gl_cfg_p in args.global_path:
        gl_cfg = OmegaConf.load(gl_cfg_p)
        global_configs = OmegaConf.merge(global_configs, gl_cfg)

    # The dir of saving work
    if not os.path.exists(global_configs.PATH) and global_configs.LOG:
        os.makedirs(global_configs.PATH)

    configs = OmegaConf.load(args.cfg_path[0])

    if no_attr(configs, "BRANCHCONTINUE"):
        configs["BRANCHCONTINUE"] = args.branch_continue
    else:
        configs["BRANCHCONTINUE"] = args.branch_continue or configs["BRANCHCONTINUE"]

    # work sleep time setting
    work_sleep = WorkSleep(sleep_interval=SLEEP_INTERVAL, sleep_time=SLEEP_TIME)

    work(configs = configs,
         path_dir = global_configs.PATH,
         paths_list = args.cfg_path,
         names_list = args.workname,
         global_configs = global_configs,
         work_sleep = work_sleep)