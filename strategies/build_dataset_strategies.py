import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import pickle
import omegaconf
from utils.get_obj import get_obj_from_str
from utils.get_attr import no_attr, from_pos_add_element_to_dict
from subwork import subwork
import numpy as np

class AutoDataset(Dataset):
    def __init__(self, configs, tag, workdata=None, self_word=["SELF"], workdata_word=["WORKDATA"]):
        self.datasetdict = {}
        self.datasettypedict = {}
        self.bridge = configs.dataset[tag]["BRIDGE"]
        self.bridge_no_prefix = self.bridge.split('.')[1]

        for subwork_key, subwork_value in configs.dataset[tag]["INIT"].items():
            func_name = subwork(dic={self_word[0]: configs, "DATASET": self.datasetdict, workdata_word[0]: workdata},
                                subwork_value=subwork_value,
                                subwork_key=subwork_key)

            if (not no_attr(subwork_value, "OUTPUT")) and (not no_attr(subwork_value, "OUTPUTtype")):
                output_len = len(subwork_value["OUTPUT"])
                for i, outtype_str in enumerate(subwork_value["OUTPUTtype"]):
                    if i>=output_len:
                        break
                    if isinstance(subwork_value["OUTPUT"][i], str):
                        output_split = subwork_value["OUTPUT"][i].split('.')
                        if output_split[0] == "DATASET":
                            output_sub_level = len(output_split) - 1
                        else:
                            output_sub_level = len(output_split)
                        if output_sub_level != 1:
                            raise Exception(f'The query level of sub-dict must be 1, but there is {subwork_value["OUTPUT"][i]} in {subwork_key}!!!')
                    try:
                        outtype = auto_dataset_output_type(outtype_str)
                    except:
                        raise Exception(f"The type of {outtype_str} is not set in the function of auto_dataset_output_type!!!")
                    if outtype is not None:
                        from_pos_add_element_to_dict(
                            dic={"DATASET": self.datasettypedict},
                            element=outtype,
                            pos=subwork_value["OUTPUT"][i]
                        )
        self.valid_keys = list(self.datasettypedict.keys())
        self.datasetdict["length"] = len(self.datasetdict[self.bridge_no_prefix])
        
    def __len__(self):
        return self.datasetdict["length"]
    
    def __getitem__(self, idx):
        return tuple(            
            (self.datasettypedict[key](self.datasetdict[key][idx]) \
             if isinstance(self.datasetdict[key], list) \
             else self.datasettypedict[key](self.datasetdict[key][self.datasetdict[self.bridge_no_prefix][idx]]) \
                for key in self.valid_keys)
            )
        
def auto_dataset_output_type(type_str):
    if type_str is None:
        return None
    elif type_str == "NOCHANGE":
        return lambda x: x
    elif type_str == "torch.LongTensor":
        return lambda x: torch.LongTensor(x)
    elif type_str == "torch.FloatTensor":
        return lambda x: torch.FloatTensor(x)
    elif type_str == "torch.BoolTensor":
        return lambda x: torch.BoolTensor(x)
