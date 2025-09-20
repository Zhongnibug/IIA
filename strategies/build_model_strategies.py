import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from utils.get_attr import no_attr
from utils.get_attr import convert_pos_to_dict_element
from utils.get_obj import get_obj_from_str
from subwork import subwork
from omegaconf import OmegaConf

def auto_build_model(model_configs, workdata=None, workdata_word=["WORKDATA"]):
    module_temp_dict = {}

    # get model checkpoint path
    load_model_dic = {}
    if not no_attr(model_configs, "load_model"):
        for module_name, module_checkpoint_path in model_configs["load_model"].items():
            while(isinstance(module_checkpoint_path, str) and module_checkpoint_path.split('.')[0] == workdata_word[0]):
                module_checkpoint_path = convert_pos_to_dict_element(
                    dic = {workdata_word[0]: workdata},
                    pos = module_checkpoint_path,
                    self_dic=None
                )
            load_model_dic[module_name] = module_checkpoint_path

    # frozen_list        
    if not no_attr(model_configs, "frozen_list"):
        frozen_list = model_configs["frozen_list"]
    else:
        frozen_list = []
    
    # build model
    model, load_model_map = auto_build_module(model_configs, "main", module_temp_dict, load_model_dic=load_model_dic, frozen_list=frozen_list)

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)

    # load model
    for load_name, load_path in load_model_dic.items():
        if load_name == "main":
            model.load_state_dict(torch.load(load_path))
        else:
            module_map = load_model_map[load_name]
            sub_module = model
            for sub_name in module_map:
                sub_module = sub_module.get_submodule(sub_name)
            sub_module.load_state_dict(torch.load(load_path))

    # torch.save(model.state_dict(), 'model_parameters_ow.pth')

    return model

def auto_build_module(model_configs, module_name, module_temp_dict, load_model_dic={}, frozen_list=[]):

    load_model_map = {}
    if not no_attr(model_configs, module_name):
        if not no_attr(model_configs[module_name], "module"):
            args_dict = {}
            if not no_attr(model_configs[module_name], "args"):
                for args_key, args_value in model_configs[module_name]["args"].items():
                    if isinstance(args_value, str) and not no_attr(model_configs[args_value], "module") and no_attr(module_temp_dict, args_value):
                        args_module, sub_load_model_map = auto_build_module(model_configs, args_value, module_temp_dict, load_model_dic=load_model_dic, frozen_list=frozen_list)
                        for load_name, load_lst in sub_load_model_map.items():
                            load_model_map[load_name] = [args_key] + load_lst
                        args_dict[args_key] = args_module
                        module_temp_dict[args_value] = args_module
                    elif isinstance(args_value, str) and not no_attr(model_configs[args_value], "module"):
                        args_dict[args_key] = copy.deepcopy(module_temp_dict[args_value])
                    elif isinstance(args_value, str):
                        args_dict[args_key] = model_configs[args_value]
                    else:
                        args_dict[args_key] = args_value
            module = get_obj_from_str(model_configs[module_name]["module"])(**args_dict)
            if module_name in load_model_dic.keys():
                load_model_map[module_name] = []
            if module_name in frozen_list:
                for param in module.parameters():
                    param.requires_grad = False
            return module, load_model_map
        else:
            raise Exception(f"The {module_name} is not a module!!!")
    else:
        raise Exception(f"The {module_name} does not exist in the model config!!!")
