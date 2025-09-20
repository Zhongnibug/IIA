import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import pickle
import omegaconf
from utils.get_obj import get_obj_from_str
from utils.get_attr import no_attr, convert_pos_to_dict_element

def auto_dataload(custom_dataset, configs, workdata, tag):
    
    function_list = []
    input_keys_list = []
    output_keys_list = []
    use_list = []
    out_generate_list = []

    non_dataset_data = {}

    sub_dic_name_list = workdata["self_word"] + workdata["workdata_word"]

    for subwork_key, subwork_value in configs.dataload[tag]["BATCH"].items():
        
        # Load function
        if no_attr(subwork_value, "FUNCTION"):
            raise Exception(f"There is no FUNCTION in {subwork_key}!!!")
        else:
            func_str = subwork_value["FUNCTION"]
            while(isinstance(func_str, str) \
                and (func_str.split('.')[0] in sub_dic_name_list)):
                func_str = convert_pos_to_dict_element(
                    dic={workdata["self_word"][0]: configs, workdata["workdata_word"][0]: workdata},
                    pos=func_str,
                    self_dic=None
                    )
            if isinstance(func_str, str):
                func = get_obj_from_str(func_str)
            else:
                func = func_str

        function_list.append(func)

        # Load input data
        input_keys_dict = {}
        for input_key, input_data in subwork_value["INPUT"].items():
            while(isinstance(input_data, str) \
                  and (input_data.split('.')[0] in sub_dic_name_list)):
                input_data = convert_pos_to_dict_element(
                    dic={workdata["self_word"][0]: configs, workdata["workdata_word"][0]: workdata},
                    pos=input_data,
                    self_dic=None
                )
            if isinstance(input_data, str) and ((input_data in custom_dataset.valid_keys) or (input_data in out_generate_list)):
                input_keys_dict[input_key] = input_data
            elif isinstance(input_data, str) and input_data.split(".")[0] == "DATASET":
                if len(input_data.split(".")) - 1 != 1:
                    raise Exception(f'In dataloader, {subwork_key}-{input_key} use the key, "DATASET", so the query level of sub-dict of it must be 1, but there is {input_data}!!!')
                elif input_data.split(".")[1] in custom_dataset.valid_keys or input_data.split(".")[1] in out_generate_list:
                    input_keys_dict[input_key] = input_data.split(".")[1]
                else:
                    raise Exception(f"In dataloader, {subwork_key}-{input_key} use the key, 'DATASET', so the query key must be in dataset's valid-keys, but there is {input_data}!!!")
            else:
                non_dataset_data[subwork_key+"_"+input_key+"_dataload"] = input_data
                input_keys_dict[input_key] = subwork_key+"_"+input_key+"_dataload"

        input_keys_list.append(input_keys_dict)

        # Load output data
        output_keys = []
        if not no_attr(subwork_value, "OUTPUT"):
            for output_key, whether_use in subwork_value["OUTPUT"].items():
                output_keys.append(output_key)
                out_generate_list.append(output_key)
                if whether_use:
                    use_list.append(output_key)
            output_keys_list.append(output_keys)          

    data_loader = DataLoader(custom_dataset,
                             batch_size=configs.dataload[tag]['batch_size'],
                             shuffle=configs.dataload[tag]['shuffle'],
                             num_workers=4,
                             collate_fn=lambda x: get_obj_from_str(
                                 configs.dataload[tag]['collate_function_strategies']
                                 )(x, configs, non_dataset_data, custom_dataset.valid_keys, 
                                   function_list, input_keys_list, output_keys_list, use_list)
                             )
    return data_loader, use_list